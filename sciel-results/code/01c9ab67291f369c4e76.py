import numpy as np
import time

class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate and LCB acquisition.
    Uses adaptive beta and local perturbations around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_min = -5.0
        self.x_max = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 10s buffer before 300s kill

        # 1. Initialization: Sample a few points
        rng = np.random.default_rng(42)
        
        # Initial points: 10 random
        n_init = 10
        X_init = rng.uniform(self.x_min, self.x_max, size=(n_init, self.dim))
        Y_init = np.array([func(x) for x in X_init])
        
        self.X = X_init
        self.Y = Y_init
        
        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()
        
        calls_made = n_init
        
        # GP Parameters
        # Length scale
        l = 0.8 
        # Signal variance
        sigma_f_sq = 1.0
        # Noise
        sigma_n_sq = 1e-6
        
        # Acquisition: LCB (Lower Confidence Bound) for minimization
        # Adaptive beta: start high for exploration, decay for exploitation
        beta_start = 3.0
        beta_end = 0.5

        def compute_kernel(X1, X2):
            # RBF kernel
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-sq_dist / (2 * l**2))

        while calls_made < self.budget:
            # Check time limit with safety margin
            if time.time() > deadline:
                break
                
            # Fit GP:
            K = compute_kernel(self.X, self.X)
            K += (sigma_n_sq + 1e-8) * np.eye(len(self.X))
            
            # Standardize Y for numerical stability
            y_mean = self.Y.mean()
            y_std = self.Y.std()
            if y_std < 1e-8: y_std = 1.0
            Y_std = (self.Y - y_mean) / y_std
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                alpha = np.linalg.inv(K) @ Y_std
            
            # Generate candidates: mix of random and local perturbations around best
            n_candidates = 500
            X_cand = rng.uniform(self.x_min, self.x_max, size=(n_candidates, self.dim))
            
            # Add local perturbations around current best (exploitation)
            # Use a larger number of local points to improve exploitation
            n_local = 300
            sigma_local = 0.5
            X_local = best_x + sigma_local * rng.standard_normal((n_local, self.dim))
            X_local = np.clip(X_local, self.x_min, self.x_max)
            X_cand = np.vstack([X_cand, X_local])
            
            # Predict on candidates
            K_s = compute_kernel(X_cand, self.X) # (M, N)
            
            # Mean: K_s @ alpha * y_std + y_mean
            mu = (K_s @ alpha) * y_std + y_mean
            
            # Variance: sigma_f_sq - K_s @ K^-1 @ K_s.T
            if 'L' in locals():
                v = np.linalg.solve(L, K_s.T) # (N, M)
                var = sigma_f_sq - np.sum(v**2, axis=0)
            else:
                # Fallback if cholesky failed
                K_inv = np.linalg.inv(K)
                v = K_inv @ K_s.T
                var = sigma_f_sq - np.sum(K_s * v, axis=1)
            
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            # Adaptive beta: decay from beta_start to beta_end over the budget
            progress = calls_made / self.budget
            beta = beta_start + (beta_end - beta_start) * progress
            
            # Acquisition: LCB = mu - beta * std
            # We want to minimize this.
            acq = mu - beta * std
            
            best_idx = np.argmin(acq)
            x_new = X_cand[best_idx]
            
            # Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # Update
            self.X = np.vstack([self.X, x_new])
            self.Y = np.append(self.Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new
                
        return best_x, best_y
