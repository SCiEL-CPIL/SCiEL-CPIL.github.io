import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)
        
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Data storage for GP
        X_data = []
        Y_data = []
        
        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            X_data.append(x.copy())
            Y_data.append(f)
            return f
        
        # Phase 1: Bayesian Optimization with GP surrogate
        # Use first ~120 calls for BO, remaining for local polish
        bo_budget = min(120, self.budget)
        
        def matern_kernel(X1, X2, length_scale=1.0, variance=1.0):
            """Matern 5/2 kernel"""
            d = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2))
            sqrt3d = np.sqrt(3.0) * d / length_scale
            K = variance * (1.0 + sqrt3d) * np.exp(-sqrt3d)
            return K
        
        def gp_predict(X_new, X_train, y_train, length_scale=1.0, variance=1.0, noise=1e-6):
            """Simple GP prediction with Matern kernel"""
            K = matern_kernel(X_train, X_train, length_scale, variance)
            K += noise * np.eye(len(X_train))
            K_s = matern_kernel(X_train, X_new, length_scale, variance)
            try:
                alpha = np.linalg.solve(K, y_train)
                mu = K_s.T @ alpha
                v = np.linalg.solve(K, K_s)
                K_ss = matern_kernel(X_new, X_new, length_scale, variance)
                var = np.diag(K_ss - v.T @ v)
                var = np.maximum(var, 1e-12)
                sigma = np.sqrt(var)
            except np.linalg.LinAlgError:
                mu = np.full(len(X_new), np.mean(y_train))
                sigma = np.full(len(X_new), np.std(y_train) + 1e-6)
            return mu, sigma
        
        def expected_improvement(x, best_y, mu, sigma):
            """Expected Improvement acquisition function"""
            improvement = best_y - mu
            with np.errstate(divide='ignore', invalid='ignore'):
                z = improvement / sigma
                ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(sigma > 1e-10, ei, 0.0)
            return np.sum(ei, axis=1) if ei.ndim > 1 else ei
        
        # Initial random points for GP
        n_init = min(10, bo_budget)
        for _ in range(n_init):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(lb, ub)
            call_func(x)
        
        # BO loop
        while calls_made < bo_budget and time.time() < deadline:
            if len(X_data) < 2:
                break
                
            X_arr = np.array(X_data)
            Y_arr = np.array(Y_data)
            best_y = np.min(Y_arr)
            
            # Generate candidate points
            n_candidates = 200
            X_cand = rng.uniform(lb, ub, size=(n_candidates, self.dim))
            
            # GP predictions
            mu, sigma = gp_predict(X_cand, X_arr, Y_arr, length_scale=1.5, variance=1.0, noise=1e-5)
            
            # EI acquisition
            ei = expected_improvement(X_cand, best_y, mu, sigma)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            # Evaluate
            f = call_func(x_next)
            
            if not np.isfinite(f):
                break
        
        # Phase 2: Local polish with L-BFGS-B from best points
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            # Try L-BFGS-B from the current best
            remaining = self.budget - calls_made
            max_fev = min(remaining, 100)
            
            try:
                res = minimize(
                    lambda x: call_func(np.array(x)),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfev': max_fev, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            except Exception:
                pass
            
            # If still have budget, try from a few other good points
            if calls_made < self.budget and time.time() < deadline and len(X_data) >= 3:
                # Sort by function value and try top 2
                indices = np.argsort(Y_data)[:min(2, len(Y_data))]
                for idx in indices:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    if np.allclose(X_data[idx], best_x, atol=1e-6):
                        continue
                    remaining = self.budget - calls_made
                    max_fev = min(remaining, 50)
                    if max_fev <= 0:
                        break
                    try:
                        res = minimize(
                            lambda x: call_func(np.array(x)),
                            np.array(X_data[idx]),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxfev': max_fev, 'ftol': 1e-15, 'gtol': 1e-15}
                        )
                    except Exception:
                        pass
        
        # Fallback
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
