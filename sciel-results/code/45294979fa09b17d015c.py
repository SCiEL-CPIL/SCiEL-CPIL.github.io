import numpy as np
import time
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    """Two-phase surrogate-assisted optimizer for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 330s limit, use 310s
        self.deadline = self.start_time + 310.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            # Clip to bounds
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Phase 1: Global exploration with Sobol sampling
        # We want to sample roughly 50 points to build a surrogate.
        # Sobol sequence gives good space-filling properties.
        n_explore = 50
        # Ensure we have enough budget
        if self.budget < n_explore + 10:
            n_explore = max(10, self.budget // 2)

        # Generate Sobol sequence
        # scipy.stats.qmc.Sobol
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # sample_size must be power of 2 for best quality, but any int works
            # We want n_explore points.
            points = sampler.random(n_explore)
            # Scale from [0, 1] to [-5, 5]
            x_explore = points * 10.0 - 5.0
        except Exception:
            # Fallback to random if qmc fails
            rng = np.random.default_rng(42)
            x_explore = rng.uniform(-5, 5, (n_explore, self.dim))

        X = np.zeros((0, self.dim))
        Y = np.zeros(0)

        # Evaluate exploration points
        for x in x_explore:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            try:
                val = wrapped_func(x)
                X = np.vstack([X, x])
                Y = np.append(Y, val)
            except StopIteration:
                break
            except Exception:
                continue

        if len(X) == 0:
            # Fallback: just run a simple local search from origin
            try:
                res = scipy.optimize.minimize(wrapped_func, np.zeros(self.dim), method='Nelder-Mead', options={'maxiter': 100})
            except:
                pass
            return self.best_x, self.best_f

        # Phase 2: Fit Gaussian Process Surrogate
        # We use a simple GP with RBF kernel.
        # To avoid numerical issues with small variances, we normalize Y.
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std

        # Kernel function: RBF + White Noise
        def kernel(X1, X2, length_scale=1.0, sigma_f=1.0, noise=0.01):
            # Compute pairwise distances
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0) # Numerical stability
            K = sigma_f**2 * np.exp(-0.5 * sq_dists / length_scale**2)
            # Add noise to diagonal
            K[np.diag_indices_from(K)] += noise
            return K

        # Optimize hyperparameters roughly (grid search or simple optimization)
        # Since n is small (50), we can afford a small grid search.
        best_ll = -np.inf
        best_ls = 1.0
        best_sf = 1.0
        best_noise = 0.01
        
        # Simple grid search for length_scale and sigma_f
        ls_candidates = [0.5, 1.0, 2.0, 5.0]
        sf_candidates = [0.5, 1.0, 2.0]
        noise_candidates = [0.01, 0.1, 1.0]

        for ls in ls_candidates:
            for sf in sf_candidates:
                for noise in noise_candidates:
                    try:
                        K = kernel(X, X, ls, sf, noise)
                        # Check positive definiteness
                        try:
                            L = np.linalg.cholesky(K)
                        except np.linalg.LinAlgError:
                            continue
                        
                        # Log likelihood
                        # log p(y|X) = -0.5 * y^T K^-1 y - 0.5 * log det(K) - n/2 log(2pi)
                        # log det(K) = 2 * sum(log(diag(L)))
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                        log_det = 2 * np.sum(np.log(np.diag(L)))
                        ll = -0.5 * np.dot(Y_norm, alpha) - 0.5 * log_det - len(Y) * 0.5 * np.log(2 * np.pi)
                        
                        if ll > best_ll:
                            best_ll = ll
                            best_ls = ls
                            best_sf = sf
                            best_noise = noise
                    except Exception:
                        continue

        # If no valid hyperparams found, use defaults
        if best_ll == -np.inf:
            best_ls = 1.0
            best_sf = 1.0
            best_noise = 0.01

        K = kernel(X, X, best_ls, best_sf, best_noise)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))

        def gp_predict_mean(x_new):
            # x_new shape: (m, dim)
            if x_new.ndim == 1:
                x_new = x_new[np.newaxis, :]
            K_s = kernel(X, x_new, best_ls, best_sf, 0.0) # No noise for prediction covariance
            # Mean = K(x_new, X) K(X, X)^-1 Y
            # K(x_new, X) is transpose of K_s
            mean = np.dot(K_s.T, alpha)
            return mean * y_std + y_mean

        # Phase 3: Use surrogate to find promising regions
        # We can either:
        # 1. Minimize the GP mean directly using a local optimizer.
        # 2. Sample points from the GP posterior and pick the best.
        # 3. Use the GP to guide a local optimizer (e.g., LBFGS or NM) start.
        
        # Let's try minimizing the GP mean using Nelder-Mead or Powell.
        # The GP mean is smooth, so gradient-based methods might work, but we don't have gradients.
        # Nelder-Mead is robust.
        
        # Generate candidate starting points for local optimization based on GP
        # 1. The global minimum of the GP mean (approximated by sampling)
        # 2. The current best point
        # 3. Points with low GP variance (exploit) and high GP uncertainty (explore) - but we want to exploit for now.
        
        # Sample 100 points to find good GP minima
        rng = np.random.default_rng(123)
        n_gp_samples = 100
        x_gp_samples = rng.uniform(-5, 5, (n_gp_samples, self.dim))
        gp_means = gp_predict_mean(x_gp_samples)
        
        # Sort by GP mean
        idx_sorted = np.argsort(gp_means)
        
        # Take top 5 candidates from GP, plus current best, plus origin
        start_candidates = []
        for i in range(min(5, len(idx_sorted))):
            start_candidates.append(x_gp_samples[idx_sorted[i]].copy())
        
        if self.best_x is not None:
            start_candidates.append(self.best_x.copy())
        start_candidates.append(np.zeros(self.dim))
        
        # Phase 4: Local Optimization with remaining budget
        # Use Nelder-Mead or Powell starting from these candidates.
        # We have budget - calls_made left.
        
        # We will run multiple local optimizations.
        # To avoid getting stuck, we limit the number of evaluations per local run.
        
        # Shuffle candidates? No, prioritize GP bests.
        
        for x0 in start_candidates:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            
            # If we are close to deadline, skip
            if self.deadline - time.time() < 5.0:
                break
            
            remaining = self.budget - self.calls_made
            if remaining < 10:
                break
            
            # Use Nelder-Mead for local refinement
            # Adaptive NM is good for Rosenbrock
            try:
                # Use a smaller maxfev to ensure we try multiple starts
                maxfev = min(remaining, 50)
                
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'maxiter': 1000,
                        'adaptive': True
                    }
                )
                
                # wrapped_func updates self.best_x and self.best_f
            except StopIteration:
                break
            except Exception:
                continue

        # Final check
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)

        return self.best_x, self.best_f
