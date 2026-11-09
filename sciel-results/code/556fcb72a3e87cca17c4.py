"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Surrogate-Assisted Optimization using Gaussian Process and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Storage for observed points
        X_obs = []
        y_obs = []
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def check_time():
            return time.time() > deadline

        def gp_predict(X_new):
            """
            Predict mean and std for X_new using GP.
            X_new: (N_new, dim)
            Returns: (mean, std)
            """
            if len(X_obs) == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0]) * 10.0
            
            X_train = np.array(X_obs)
            y_train = np.array(y_obs)
            
            # Standardize y for numerical stability
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y_train - y_mean) / y_std
            
            # Matern kernel with length scales
            # Hyperparameters: length_scale, variance, noise
            # For simplicity and speed in 5D, we use fixed reasonable hyperparameters
            # or a very simple optimization. Given budget 500, we can't do heavy HPO.
            # Let's use a Matern-5/2 kernel.
            
            def matern52(A, L):
                # A is (N, M) or (N, N)
                # Compute distance matrix
                d = np.sqrt(np.sum((A[:, None, :] - A[None, :, :])**2, axis=-1))
                zd = d / L
                return np.exp(-zd) * (1 + zd + zd**2 / 3)
            
            # Fixed hyperparameters tuned for [-5,5]^5
            # Length scale ~1.0, Variance ~1.0, Noise ~1e-6
            L = 1.0
            v = 1.0
            noise = 1e-8
            
            K = matern52(X_train, L) * v
            K[np.diag_indices_from(K)] += noise
            
            K_star = matern52(X_new, L) * v
            
            # Solve K alpha = y_norm
            try:
                L_chol = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_norm))
            except:
                # Fallback to least squares if singular
                alpha = np.linalg.lstsq(K, y_norm, rcond=None)[0]
                
            mu_star = K_star @ alpha
            v_star = v - K_star @ alpha
            var_star = np.diag(v_star)
            
            # Transform back
            mu = mu_star * y_std + y_mean
            std = np.sqrt(np.maximum(var_star, 1e-10)) * y_std
            
            return mu, std

        def expected_improvement(mu, std, best_y, xi=0.0):
            """
            Calculate EI for each point.
            mu, std: arrays
            best_y: current best function value (minimization)
            """
            z = (best_y - xi - mu) / std
            ei = (best_y - xi - mu) * norm.cdf(z) + std * norm.pdf(z)
            # Handle numerical issues where std is 0
            ei[std < 1e-8] = 0.0
            return ei

        # Phase 1: Initial Sampling (Latin Hypercube or Random)
        # We need enough points to build a GP. 10-20 points is good for 5D.
        n_init = 15
        init_budget = min(n_init, budget)
        
        # Generate initial points
        # Use a mix of center and random to cover the space
        # Simple random is robust.
        X_init = rng.uniform(-5, 5, size=(init_budget, dim))
        
        for i in range(init_budget):
            if calls_used >= budget or check_time():
                break
            x = X_init[i]
            f = func(x)
            X_obs.append(x)
            y_obs.append(f)
            calls_used += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Surrogate Assisted Optimization
        # Loop until budget or time runs out
        # We can afford a few hundred GP evaluations.
        # To speed up EI maximization, we use a coarse grid + local refinement.
        
        while calls_used < budget and not check_time():
            # 1. Predict on a candidate set
            # Generate a coarse grid for exploration
            # In 5D, a full grid is expensive. Use random candidates + local around best.
            
            # Candidate pool:
            # a) 500 random points in [-5, 5]^5
            # b) 100 points around current best (sigma=0.5)
            # c) 100 points around best 3 points (sigma=1.0)
            
            n_rand = 500
            n_local_best = 100
            n_local_top = 100
            
            X_cand = rng.uniform(-5, 5, size=(n_rand, dim))
            
            # Local around best
            if best_x is not None:
                X_local1 = best_x[None, :] + rng.normal(0, 0.5, size=(n_local_best, dim))
                X_local1 = np.clip(X_local1, -5, 5)
                X_cand = np.vstack([X_cand, X_local1])
                
                # Local around other good points
                # Sort observed by fitness
                if len(y_obs) > 3:
                    y_arr = np.array(y_obs)
                    x_arr = np.array(X_obs)
                    top_indices = np.argsort(y_arr)[:3]
                    for idx in top_indices:
                        X_local2 = x_arr[idx][None, :] + rng.normal(0, 1.0, size=(n_local_top, dim))
                        X_local2 = np.clip(X_local2, -5, 5)
                        X_cand = np.vstack([X_cand, X_local2])
            
            # 2. Predict mean and std on candidates
            mu, std = gp_predict(X_cand)
            
            # 3. Calculate EI
            ei = expected_improvement(mu, std, best_f)
            
            # 4. Select best candidate
            idx_best_cand = np.argmax(ei)
            x_next = X_cand[idx_best_cand]
            
            # 5. Evaluate
            f_next = func(x_next)
            X_obs.append(x_next)
            y_obs.append(f_next)
            calls_used += 1
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
            
            # Optimization: If we have very few calls left, stop
            if calls_used >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
