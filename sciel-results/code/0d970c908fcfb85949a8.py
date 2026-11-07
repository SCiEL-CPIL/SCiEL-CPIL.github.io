import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0 # Safety margin
        n = self.dim
        B = self.budget
        
        # State
        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0
        X = [] # List of points
        Y = [] # List of values
        
        def time_ok():
            return (time.time() - start) < deadline

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B:
                return float('inf')
            calls += 1
            y = np.clip(x, -5, 5)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            X.append(y.copy())
            Y.append(f)
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling
        # Use Latin Hypercube or random sampling
        n_init = min(30, B)
        for i in range(n_init):
            if not time_ok() or calls >= B:
                break
            x = rng.uniform(-5, 5, size=n)
            ev(x)
            
        if not X:
            return best_x, best_f

        X = np.array(X)
        Y = np.array(Y)
        
        # GP Parameters
        # Normalize Y
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-10:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std
        
        # Kernel: RBF with fixed length scale
        # We can adjust length scale based on dimension, e.g., 1.0
        length_scale = 1.0
        noise = 1e-8
        
        def kernel(X1, X2):
            # X1: (N, n), X2: (M, n)
            # dist^2 = sum((X1 - X2)^2, axis=1)
            sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (length_scale**2))
        
        def fit_gp(X_train, Y_train):
            K = kernel(X_train, X_train)
            K += noise * np.eye(len(X_train))
            # Solve for alpha
            try:
                alpha = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
            return alpha, K
        
        def predict_gp(x_new, alpha, X_train, Y_train):
            # x_new: (n,) or (M, n)
            if x_new.ndim == 1:
                x_new = x_new.reshape(1, -1)
            K_s = kernel(X_train, x_new) # (N, M)
            K = kernel(X_train, X_train) # (N, N)
            K += noise * np.eye(len(X_train))
            
            # Mean
            mu = K_s.T @ alpha # (M,)
            
            # Variance
            # Var = k(x, x) - k(x, X)^T K^{-1} k(X, x)
            # We already have K^{-1} alpha, but we need K^{-1} k(X, x)
            # Let's compute K_inv = np.linalg.inv(K) or solve
            # For efficiency, we can precompute K_inv
            # But we recompute K every time? No, X_train changes.
            # Let's compute K_inv in fit_gp and store it.
            # Actually, for small N, inv is fine.
            
            return mu, None # Variance calculation is tricky without K_inv stored
        
        # Better approach: Store K_inv
        class GPM:
            def __init__(self, X, Y, ls, noise):
                self.X = X
                self.Y = Y
                self.ls = ls
                self.noise = noise
                self.K = kernel(X, X)
                self.K += noise * np.eye(len(X))
                self.K_inv = np.linalg.inv(self.K)
                self.alpha = self.K_inv @ Y
                
            def predict(self, x_new):
                if x_new.ndim == 1:
                    x_new = x_new.reshape(1, -1)
                K_s = kernel(self.X, x_new) # (N, M)
                mu = K_s.T @ self.alpha
                # Var = 1 - K_s.T @ K_inv @ K_s
                v = K_s.T @ self.K_inv @ K_s
                var = 1.0 - np.diag(v)
                var = np.maximum(var, 0.0) # Numerical stability
                std = np.sqrt(var)
                return mu, std

        # EI Function
        def ei(x, gp_model, best_y_norm):
            mu, std = gp_model.predict(x)
            if std < 1e-10:
                return 0.0
            z = (best_y_norm - mu) / std
            # Normal CDF and PDF
            from scipy.stats import norm
            ei_val = (best_y_norm - mu) * norm.cdf(z) + std * norm.pdf(z)
            return ei_val

        # Phase 2: Surrogate-Guided Search
        # We will use a mix of EI maximization and local polish
        
        # To speed up EI maximization, we can use a grid + local refinement
        # Or just multi-start Nelder-Mead on -EI
        
        from scipy.stats import norm
        
        # Precompute norm functions if possible? No, they are fast enough.
        
        surrogate_iterations = 0
        max_surrogate_iters = B - len(X)
        
        # Switch to local polish if we are close to the end or if EI is low
        local_polish_start = 0.7 * B
        
        while calls < B and time_ok():
            if len(X) >= local_polish_start:
                # Switch to Local Polish
                break
                
            # Fit GP
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr)
            if y_std < 1e-10:
                y_std = 1.0
            Y_norm = (Y_arr - y_mean) / y_std
            best_y_norm = (best_f - y_mean) / y_std
            
            gp_model = GPM(X_arr, Y_norm, length_scale, noise)
            
            # Maximize EI
            # Multi-start local search
            best_ei_val = -np.inf
            best_x_ei = None
            
            # Start points: random, best_x, center
            start_points = [
                best_x.copy(),
                np.zeros(n),
                rng.uniform(-5, 5, size=n),
                rng.uniform(-5, 5, size=n),
                rng.uniform(-5, 5, size=n)
            ]
            
            for sp in start_points:
                if not time_ok():
                    break
                # Optimize -EI
                def neg_ei(x):
                    x = np.clip(x, -5, 5)
                    return -ei(x, gp_model, best_y_norm)
                
                res = minimize(neg_ei, sp, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_ei_val:
                    best_ei_val = res.fun
                    best_x_ei = res.x
            
            if best_x_ei is None:
                break
                
            # Evaluate
            f_new = ev(best_x_ei)
            surrogate_iterations += 1
            
            # If no improvement for a while, maybe adjust strategy?
            # For now, just continue.
            
        # Phase 3: Local Polish
        if calls < B and time_ok():
            # Use Nelder-Mead or CMA-ES
            # Nelder-Mead is simple
            x0 = best_x.copy()
            
            # Initial simplex
            simplex = [x0]
            for i in range(n):
                pt = x0.copy()
                # Step size: 10% of range or 0.5
                step = 0.5
                if i % 2 == 0:
                    pt[i] += step
                else:
                    pt[i] -= step
                simplex.append(pt)
            
            # Simple Nelder-Mead implementation or use scipy
            def obj(x):
                if not time_ok() or calls >= B:
                    return float('inf')
                return ev(x)
            
            # Use scipy minimize
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            # Note: scipy minimize might call obj multiple times.
            # We need to ensure we don't exceed budget.
            # The `ev` function checks budget and time.
            
        return best_x, best_f
