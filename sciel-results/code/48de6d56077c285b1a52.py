"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization using Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize with random points
        n_init = 10
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(Y)]
        best_fx = np.min(Y)
        call_count = n_init

        # GP Parameters
        l = 1.0
        sigma_f = 1.0
        noise = 1e-5

        def kernel(x1, x2):
            # x1: (n, d), x2: (m, d)
            sq_dists = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2 * x1 @ x2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (l**2))

        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Normalize Y
            y_mean = Y.mean()
            y_std = Y.std()
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y - y_mean) / y_std
            best_fx_norm = (best_fx - y_mean) / y_std
            
            # Pre-compute GP terms
            K_train = kernel(X, X)
            K_train += noise * np.eye(len(X))
            try:
                K_inv = np.linalg.inv(K_train)
                alpha = K_inv @ Y_norm
            except np.linalg.LinAlgError:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5, 5, size=dim)
                fx_next = func(x_next)
                call_count += 1
                X = np.vstack([X, x_next])
                Y = np.append(Y, fx_next)
                if fx_next < best_fx:
                    best_fx = fx_next
                    best_x = x_next.copy()
                continue

            def expected_improvement(x):
                # x is a single point (dim,)
                x = np.array([x])
                K_star = kernel(x, X) # (1, N)
                mu = K_star @ alpha
                K_star_star = kernel(x, x)
                var = K_star_star[0, 0] - K_star @ (K_inv @ K_star.T)
                var = max(var, 1e-10)
                std = np.sqrt(var)
                
                if std < 1e-8:
                    return 0.0
                
                z = (mu - best_fx_norm) / std
                ei = (mu - best_fx_norm) * norm.cdf(z) + std * norm.pdf(z)
                return ei

            # Optimize acquisition function
            best_acq_val = -np.inf
            best_x_next = None
            
            n_starts = 5
            for _ in range(n_starts):
                x0 = rng.uniform(-5, 5, size=dim)
                
                def acq_func(x):
                    return -expected_improvement(x)
                
                try:
                    res = minimize(acq_func, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_x_next = res.x
                except Exception:
                    pass
            
            if best_x_next is None:
                best_x_next = rng.uniform(-5, 5, size=dim)
            
            best_x_next = np.clip(best_x_next, -5, 5)
            
            fx_next = func(best_x_next)
            call_count += 1
            
            X = np.vstack([X, best_x_next])
            Y = np.append(Y, fx_next)
            
            if fx_next < best_fx:
                best_fx = fx_next
                best_x = best_x_next.copy()
                
        return best_x, best_fx

# EVOLVE-BLOCK-END
