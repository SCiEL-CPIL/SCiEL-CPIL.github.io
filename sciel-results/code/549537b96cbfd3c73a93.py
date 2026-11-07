import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
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
        rng = np.random.default_rng(42)
        
        # Data storage for GP
        X_data = []
        Y_data = []

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f
            if calls_made >= self.budget:
                return best_f
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=float)
            return f_val

        # Phase 1: Bayesian Optimization
        # 1. Initialize with Latin Hypercube sampling
        n_init = 10
        X_init = rng.uniform(-5, 5, (n_init, self.dim))
        # Actually, let's use a simple random or LHS. LHS is better.
        # Simple LHS implementation
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            X_init[:, i] = (perm + 0.5) / n_init
            X_init[:, i] = X_init[:, i] * 10 - 5
        
        for x in X_init:
            f_val = wrapper(x)
            X_data.append(x)
            Y_data.append(f_val)
            
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)
        
        # GP Hyperparameters
        # Matern 5/2 kernel
        # K(x, x') = sigma_f^2 * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
        # d = ||x - x'|| / l
        sigma_f = 1.0
        length_scale = 1.0
        sigma_n = 1e-6 # Noise
        
        def kernel(X1, X2):
            # X1: (n1, dim), X2: (n2, dim)
            # Compute squared distances
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1.x2
            sq_norms1 = np.sum(X1**2, axis=1)[:, np.newaxis]
            sq_norms2 = np.sum(X2**2, axis=1)[np.newaxis, :]
            sq_dists = sq_norms1 + sq_norms2 - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            dists = np.sqrt(sq_dists)
            
            d = dists / length_scale
            K = sigma_f**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)
            return K

        def fit_gp(X, Y):
            # Add jitter for numerical stability
            K = kernel(X, X)
            K[np.diag_indices_from(K)] += sigma_n + 1e-8
            
            try:
                L = cholesky(K, lower=True)
            except Exception:
                # If Cholesky fails, add more jitter
                K[np.diag_indices_from(K)] += 1e-6
                L = cholesky(K, lower=True)
            
            alpha = cho_solve((L, True), Y)
            return L, alpha

        def predict_gp(X_new, L, alpha, X_train):
            K_star = kernel(X_train, X_new)
            mu = K_star.T @ alpha
            K_star_star = kernel(X_new, X_new)
            cov = K_star_star - K_star.T @ cho_solve((L, True), K_star)
            var = np.diag(cov)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, y_best):
            eps = 1e-8
            imp = y_best - mu
            sigma = np.sqrt(var)
            z = imp / (sigma + eps)
            from scipy.stats import norm
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Optimization of Acquisition Function
        # We want to maximize EI, so we minimize -EI
        # We use multi-start for the acquisition optimization to avoid local minima in the acquisition landscape
        
        n_bo_iter = 40 # Total BO iterations
        # We have used n_init calls. Remaining budget: budget - n_init
        
        for i in range(n_bo_iter):
            if calls_made >= self.budget - 50: # Leave 50 for polishing
                break
            if time.time() - start_time > time_limit - 30:
                break
                
            L, alpha = fit_gp(X_data, Y_data)
            y_best = np.min(Y_data)
            
            # Define negative EI
            def neg_ei(x):
                x = x.reshape(1, -1)
                mu, var = predict_gp(x, L, alpha, X_data)
                ei = expected_improvement(mu[0], var[0], y_best)
                return -ei[0]
            
            # Multi-start for acquisition optimization
            best_acq_x = None
            best_acq_val = float('inf')
            
            # Starts: best point, random points, points near best
            starts = [best_x.copy()]
            for _ in range(5):
                starts.append(rng.uniform(-5, 5, self.dim))
            for _ in range(5):
                starts.append(np.clip(best_x + rng.normal(0, 0.5, self.dim), -5, 5))
                
            for start in starts:
                try:
                    res = minimize(
                        neg_ei,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 50}
                    )
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_acq_x = res.x
                except Exception:
                    continue
                    
            if best_acq_x is None:
                best_acq_x = rng.uniform(-5, 5, self.dim)
                
            f_val = wrapper(best_acq_x)
            X_data = np.vstack([X_data, best_acq_x])
            Y_data = np.append(Y_data, f_val)

        # Phase 2: Local Polish
        # Use the best point found by BO
        if calls_made < self.budget - 5 and best_x is not None:
            try:
                res_lbfgsb = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'gtol': 1e-14, 'ftol': 1e-14}
                )
                if res_lbfgsb.fun < best_f:
                    best_f = res_lbfgsb.fun
                    best_x = np.array(res_lbfgsb.x, dtype=float)
            except Exception:
                pass

            # Optional: Nelder-Mead if L-BFGS-B didn't help much or for robustness
            if calls_made < self.budget - 5:
                try:
                    res_nm = minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = np.array(res_nm.x, dtype=float)
                except Exception:
                    pass

        return best_x, best_f
