import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # GP Hyperparameters
        l = 1.0
        sigma_f = 1.0
        sigma_n = 1e-4
        
        # Data storage
        X = []
        Y = []
        
        def rbf_kernel(X1, X2):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (l**2))
        
        # Initial sampling
        n_init = 10
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            y = func(x)
            X.append(x)
            Y.append(y)
            
        f_best = min(Y)
        
        # Main loop
        calls_made = n_init
        while calls_made < budget:
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Compute GP parameters
            K = rbf_kernel(X_arr, X_arr) + sigma_n**2 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(len(X))
                L = np.linalg.cholesky(K)
            
            v = np.linalg.solve(L, Y_arr)
            alpha = np.linalg.solve(L.T, v)
            
            def ei_func(x):
                x = np.array(x).reshape(1, -1)
                k_x = rbf_kernel(x, X_arr).flatten() # (N,)
                
                # Mean
                mu = k_x @ alpha
                # Var
                u = np.linalg.solve(L, k_x)
                var = sigma_f**2 - np.dot(u, u)
                var = max(var, 1e-12)
                sigma = np.sqrt(var)
                
                if sigma < 1e-12:
                    return 0.0
                z = (f_best - mu) / sigma
                ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            best_x_acq = None
            best_ei_val = np.inf
            
            starts = []
            starts.append(np.array(X[np.argmin(Y)]))
            for _ in range(5):
                starts.append(rng.uniform(-5, 5, dim))
            
            for start in starts:
                start = np.clip(start, -5, 5)
                try:
                    res = minimize(
                        ei_func, 
                        start, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-6}
                    )
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_x_acq = res.x
                except Exception:
                    continue
            
            if best_x_acq is None:
                best_x_acq = rng.uniform(-5, 5, dim)
            
            best_x_acq = np.clip(best_x_acq, -5, 5)
            
            y = func(best_x_acq)
            X.append(best_x_acq)
            Y.append(y)
            calls_made += 1
            
            if y < f_best:
                f_best = y
        
        return X[np.argmin(Y)], f_best
