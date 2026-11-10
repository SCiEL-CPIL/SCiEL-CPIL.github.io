import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Normalize inputs to [0, 1] for GP
        def normalize_x(x):
            return (x - lo) / range_
        
        def denormalize_x(x_norm):
            return x_norm * range_ + lo

        # Initialize
        n_init = 10
        X = np.random.uniform(0, 1, (n_init, self.dim))
        X = denormalize_x(X)
        F = np.array([problem(x) for x in X])
        
        # Weights for weighted sum approach
        # Covering the trade-off space
        weights = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9]
        ])
        
        used = n_init
        weight_idx = 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select weight
            w = weights[weight_idx % len(weights)]
            weight_idx += 1
            
            # Compute weighted sum of objectives
            # Note: We want to MINIMIZE objectives.
            # Weighted sum: g(x) = w[0]*f1(x) + w[1]*f2(x)
            g = F @ w
            
            # Normalize g for GP
            g_mean = np.mean(g)
            g_std = np.std(g)
            if g_std == 0: g_std = 1.0
            g_norm = (g - g_mean) / g_std
            
            X_norm = normalize_x(X)
            
            # Fit GP
            # Simple GP with RBF kernel
            # K(x, x') = exp(-0.5 * ||x - x'||^2 / sigma^2)
            # We use sigma = 1.0 in normalized space
            sigma = 1.0
            sigma_f = 1.0
            
            def kernel(X1, X2):
                # X1: (n1, d), X2: (n2, d)
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                return np.exp(-0.5 * sq_dists / (sigma**2))
            
            K = kernel(X_norm, X_norm) + 1e-8 * np.eye(len(X))
            
            # Solve for GP parameters
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, g_norm))
            except np.linalg.LinAlgError:
                # Fallback to simple random search if GP fails
                x_new = np.random.uniform(0, 1, self.dim)
                x_new = denormalize_x(x_new)
                f_new = problem(x_new)
                X = np.vstack([X, x_new])
                F = np.vstack([F, f_new])
                used += 1
                continue
            
            # Acquisition Function: Expected Improvement (EI)
            # We want to minimize g.
            # f_best = min(g)
            f_best = np.min(g)
            f_best_norm = (f_best - g_mean) / g_std
            
            def ei(x_norm):
                x_norm = np.array([x_norm])
                K_s = kernel(X_norm, x_norm) # (n, 1)
                mu = K_s.T @ alpha # (1,)
                v = kernel(x_norm, x_norm) # (1, 1)
                var = sigma_f**2 - np.sum(K_s**2, axis=0)
                std = np.sqrt(np.maximum(var, 1e-8))
                
                # EI for minimization
                # Standard EI is for maximization.
                # For minimization, EI(x) = E[max(f_best - f(x), 0)]
                # Let Y ~ N(mu, std). We want E[max(f_best_norm - Y, 0)]
                # This is equivalent to maximizing E[max(Y - f_best_norm, 0)] with sign flipped?
                # No, let's derive.
                # Let Z = f_best_norm - Y. Z ~ N(f_best_norm - mu, std).
                # EI = E[max(Z, 0)]
                # E[max(Z, 0)] = (E[Z]) * Phi(E[Z]/std) + std * phi(E[Z]/std)
                mean_z = f_best_norm - mu[0]
                ei_val = mean_z * norm.cdf(mean_z / std) + std * norm.pdf(mean_z / std)
                return -ei_val[0] # Minimize negative EI (maximize EI)
            
            # Maximize EI using Nelder-Mead
            # Start from a few random points
            best_x = None
            best_ei = np.inf
            
            for _ in range(5): # 5 restarts
                x0 = np.random.uniform(0, 1, self.dim)
                res = minimize(ei, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_ei:
                    best_ei = res.fun
                    best_x = res.x
                    
            if best_x is None:
                best_x = np.random.uniform(0, 1, self.dim)
                
            x_new = denormalize_x(best_x)
            # Clip to bounds just in case
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            used += 1
            
        return X
