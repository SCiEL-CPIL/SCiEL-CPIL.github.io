import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300 - 30  # 270s safety margin
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Storage for surrogate model
        X_obs = []
        Y_obs = []
        
        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            
            if np.isnan(f_val) or np.isinf(f_val):
                return best_f
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=float)
            
            X_obs.append(np.array(x, dtype=float))
            Y_obs.append(f_val)
            
            return f_val

        # Phase 1: Initial Exploration
        # We need enough points to fit a surrogate. 
        # Let's use ~50 calls for initial random/LHS sampling.
        # Latin Hypercube is good for spread.
        n_init = 50
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # Simple LHS implementation
        def latin_hypercube(n, d):
            # n samples in d dimensions
            points = np.zeros((n, d))
            for i in range(n):
                # Shuffle indices for each dimension
                idx = np.arange(n)
                rng.shuffle(idx)
                # Actually, standard LHS: for each dimension, permute the n intervals
                # Let's do it properly
                pass
            
            # Simpler approach: 
            # For each dimension, create n intervals [i/n, (i+1)/n)
            # Sample one point in each interval for each dimension
            # Then shuffle the pairing
            
            # Standard LHS:
            # 1. Create a matrix where row i has 1 in column i (identity)
            # 2. For each column (dimension), permute the rows
            # 3. Add random noise within the cell
            
            cells = np.zeros((n, d))
            for i in range(n):
                cells[i, i] = 1.0
            
            # Permute rows for each column
            for j in range(d):
                perm = rng.permutation(n)
                # This logic is tricky to implement quickly without a library.
                # Let's just use random uniform for now, or a simple grid.
                # Actually, random uniform is fine for 5D with 50 points if we want simplicity.
                # But LHS is better. Let's use a simple stratified sampling.
                pass
            
            # Let's just use random uniform for simplicity and robustness, 
            # or a coarse grid if dim is small.
            # With dim=5, 50 random points is decent coverage.
            return rng.uniform(-5, 5, (n, d))

        X_init = latin_hypercube(n_init, self.dim)
        
        # Evaluate initial points
        for i in range(n_init):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            try:
                wrapper(X_init[i])
            except (TimeoutError, RuntimeError):
                break

        if best_x is None:
            # Should not happen, but safeguard
            best_x = np.zeros(self.dim)
            best_f = float('inf')
        
        # Phase 2: Surrogate-Assisted Optimization (Bayesian Optimization)
        # Use Gaussian Process with Expected Improvement
        # We will iterate until budget or time is up.
        
        # GP Parameters
        # We need to fit a GP. Since we don't have a fast GP library in std/scipy,
        # we will implement a simple GP with a Matern kernel or use RBF.
        # For 5D, RBF is reasonable.
        
        # Hyperparameters: length_scale, noise, variance
        # We can fix them or optimize. Fixing is faster.
        # Let's use fixed hyperparameters to avoid expensive hyperparam optimization.
        # length_scale = 1.0, noise = 1e-6, variance = 1.0 (normalized Y)
        
        # Normalize Y for GP stability
        # We'll normalize Y to [0, 1] or mean 0 var 1
        
        def fit_gp(X, Y, length_scale=1.0, noise=1e-8, var=1.0):
            n = len(X)
            d = X.shape[1]
            
            # Normalize X to [0, 1] for better conditioning?
            # Or keep as is. Let's normalize X to [0, 1] range [-5, 5]
            X_min = -5.0
            X_max = 5.0
            X_norm = (X - X_min) / (X_max - X_min)
            
            # Normalize Y
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            if Y_std < 1e-10:
                Y_std = 1.0
            Y_norm = (Y - Y_mean) / Y_std
            
            # Kernel: RBF
            # K(x, x') = var * exp(-0.5 * ||x - x'||^2 / length_scale^2)
            # Compute pairwise distances
            # X_norm is n x d
            # dist^2 = sum((x_i - x_j)^2)
            # Use broadcasting
            # (n, 1, d) - (1, n, d) -> (n, n, d)
            diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            K = var * np.exp(-0.5 * sq_dist / (length_scale**2))
            K += noise * np.eye(n)
            
            # Solve K alpha = Y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                # Fallback to lstsq
                alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
            
            return X_norm, Y_norm, Y_mean, Y_std, length_scale, noise, var, L, alpha

        def predict_gp(X_new, X_train, Y_train, length_scale, noise, var, L, alpha, Y_mean, Y_std):
            # X_new is (m, d)
            # X_train is (n, d) - normalized
            X_new_norm = (X_new - (-5.0)) / 10.0
            
            # Compute K(X_new, X_train)
            diff = X_new_norm[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            K_s = var * np.exp(-0.5 * sq_dist / (length_scale**2))
            
            # Mean: K_s @ alpha
            mu = K_s @ alpha
            
            # Variance: var - K_s @ L^{-T} @ L^{-1} @ K_s.T
            # v = L^{-1} @ K_s.T
            v = np.linalg.solve(L, K_s.T)
            var_pred = var - np.sum(v**2, axis=1)
            
            # Unnormalize
            mu = mu * Y_std + Y_mean
            var_pred = var_pred * Y_std**2
            
            # Clip variance to be positive
            var_pred = np.maximum(var_pred, 1e-10)
            std_pred = np.sqrt(var_pred)
            
            return mu, std_pred

        # Expected Improvement
        def expected_improvement(mu, sigma, best_f, xi=0.0):
            # mu, sigma are arrays
            # best_f is scalar
            # If sigma is very small, EI is 0
            sigma = np.maximum(sigma, 1e-8)
            imp = best_f - mu - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            # If imp is negative, EI should be 0? 
            # Actually, EI is defined as E[max(f_best - f(x), 0)]
            # The formula above handles this correctly if we consider the standard normal CDF.
            # However, numerically, if imp < 0 and z is very negative, cdf(z) ~ 0, pdf(z) ~ 0, so ei ~ 0.
            # But to be safe:
            ei = np.where(imp > 0, ei, 0.0)
            return ei

        # Main BO Loop
        # We will try to acquire points one by one, or in small batches.
        # Single point acquisition is safer for budget control.
        
        # Initial GP fit
        try:
            X_train_norm, Y_train_norm, Y_mean, Y_std, ls, noise, var, L, alpha = fit_gp(
                np.array(X_obs), np.array(Y_obs)
            )
        except Exception:
            X_train_norm = None
        
        # Candidate points for acquisition
        # We can use random candidates and pick the best EI.
        n_candidates = 100
        
        while calls_made < self.budget and (time.time() - start_time) < time_limit:
            # Refit GP periodically? Or every time?
            # Refitting every time is O(n^3) which might be slow for n=500.
            # 500^3 = 125M operations. Might be too slow in Python/NumPy loop.
            # Let's refit every 10-20 iterations.
            
            if X_train_norm is not None and len(X_obs) % 10 == 0:
                try:
                    X_train_norm, Y_train_norm, Y_mean, Y_std, ls, noise, var, L, alpha = fit_gp(
                        np.array(X_obs), np.array(Y_obs)
                    )
                except Exception:
                    pass
            
            # Generate candidates
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))
            
            if X_train_norm is not None:
                try:
                    mu, sigma = predict_gp(
                        X_cand, X_train_norm, Y_train_norm, ls, noise, var, L, alpha, Y_mean, Y_std
                    )
                    ei = expected_improvement(mu, sigma, best_f)
                    idx = np.argmax(ei)
                    x_next = X_cand[idx]
                except Exception:
                    # Fallback to random
                    x_next = rng.uniform(-5, 5, self.dim)
            else:
                x_next = rng.uniform(-5, 5, self.dim)
            
            # Evaluate
            try:
                wrapper(x_next)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                break
            
            # Update best
            # wrapper updates best_x, best_f
            
            # Check if we should stop early if we've converged?
            # Hard to know. Let's just run until budget.
        
        # Phase 3: Local Polish with L-BFGS-B
        # Use remaining budget for high-precision convergence
        if calls_made < self.budget and (time.time() - start_time) < time_limit:
            try:
                res_lbfgsb = minimize(
                    wrapper, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-8}
                )
                if res_lbfgsb.fun < best_f:
                    best_f = res_lbfgsb.fun
                    best_x = np.array(res_lbfgsb.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead Polish (optional, if budget remains)
        if calls_made < self.budget and (time.time() - start_time) < time_limit:
            try:
                res_nm = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.array(res_nm.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
