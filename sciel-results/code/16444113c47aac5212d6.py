import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Data storage
        X = []
        Y = []

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            x_arr = np.asarray(x, dtype=np.float64)
            f = float(func(x_arr))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_arr.copy()
            X.append(x_arr)
            Y.append(f)
            return f

        # Kernel function
        def rbf_kernel(X1, X2, length_scale=1.0, noise=1e-6):
            # Squared Euclidean distance
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0) # numerical stability
            return np.exp(-0.5 * sq_dists / (length_scale**2))

        # GP Prediction
        def gp_predict(X_new, X_train, Y_train, length_scale=1.0, noise=1e-6):
            K = rbf_kernel(X_train, X_train, length_scale)
            K += noise * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            K_star = rbf_kernel(X_new, X_train, length_scale)
            v = np.linalg.solve(L, K_star.T)
            mean = (K_star @ alpha).T
            
            K_ss = rbf_kernel(X_new, X_new, length_scale)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var)
            
            return mean, std

        # Expected Improvement Acquisition
        def expected_improvement(X_new, best_f, length_scale=1.0, noise=1e-6):
            mean, std = gp_predict(X_new, X, Y, length_scale, noise)
            Z = (mean - best_f) / std
            ei = (mean - best_f) * norm.cdf(Z) + std * norm.pdf(Z)
            return -ei # Negative because we minimize

        # Optimize acquisition function
        def optimize_acquisition(best_f, length_scale=1.0, noise=1e-6):
            # Use a few random starts for the acquisition optimizer
            best_x_acq = None
            best_ei = -np.inf
            
            for _ in range(10):
                x0 = rng.uniform(-5, 5, d)
                res = minimize(
                    expected_improvement, 
                    x0, 
                    args=(best_f, length_scale, noise),
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 50}
                )
                if res.fun < best_ei:
                    best_ei = res.fun
                    best_x_acq = res.x
            
            return best_x_acq

        # Phase 1: Initial Sampling
        n_init = min(20, budget)
        # Use a mix of random and grid-like points to ensure coverage
        # Simple random is fine for 5D
        for _ in range(n_init):
            x = rng.uniform(-5, 5, d)
            try:
                safe_func(x)
            except StopIteration:
                break

        # Phase 2: Bayesian Optimization Loop
        # We need to adapt length_scale or keep it fixed. Fixed is safer for speed.
        # Let's try a fixed length_scale of 1.0 initially.
        # If the function is very rugged, smaller length_scale might be better.
        # Schwefel is rugged. Let's try 0.5.
        
        length_scale = 0.8
        noise = 1e-6
        
        # If we ran out of budget in init, return
        if calls >= budget:
            return best_x, best_f
            
        while calls < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Fit GP and optimize acquisition
            # To speed up, we might skip GP if X is very small, but we have >= 20.
            try:
                x_next = optimize_acquisition(best_f, length_scale, noise)
            except Exception:
                # Fallback to random if BO fails
                x_next = rng.uniform(-5, 5, d)
                
            try:
                f_next = safe_func(x_next)
            except StopIteration:
                break
            
            # Adaptive length scale? 
            # If improvement is small, we might be in a local basin.
            # If we haven't improved in last 20 steps, restart or random jump?
            # For now, keep it simple.
            
        # Phase 3: Local Polish
        # If we have budget left, do a (1+1)-ES polish
        if best_x is not None and calls < budget and time.time() < deadline - 2.0:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 50
            
            while calls < budget and time.time() < deadline - 1.0 and no_improve < max_no_improve:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                # Clip to bounds
                x_candidate = np.clip(x_candidate, -5, 5)
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                    
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    
                    if sigma < 1e-6:
                        break

        return best_x, best_f
