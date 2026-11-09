"""Surrogate-assisted optimizer using Gaussian Process + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol

class Optimizer:
    """GP surrogate with Expected Improvement acquisition and adaptive hyperparameters."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization with Sobol sequence
        n_init = min(20, budget)
        sampler = Sobol(d=dim, scramble=True, seed=42)
        X = sampler.random(n_init) * 10 - 5
        Y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]
        
        # GP Hyperparameters
        length_scale = 1.0
        noise_var = 1e-3
        signal_var = 1.0
        
        # Store data
        X_train = X.copy()
        Y_train = Y.copy()
        
        # Precompute GP internals
        def kernel(A, B, ls):
            """Squared exponential kernel."""
            diff = A[:, None, :] - B[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return signal_var * np.exp(-sq_dist / (2 * ls ** 2))

        def fit_gp(X_train, Y_train, ls):
            """Fit GP and return L, alpha, y_mean, y_std."""
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y_train - y_mean) / y_std
            n = len(X_train)
            
            K = kernel(X_train, X_train, ls)
            K += noise_var * np.eye(n)
            K += 1e-8 * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                
            return L, alpha, y_mean, y_std

        def update_gp_hyperparams(X_train, Y_train):
            """Simple grid search for length scale."""
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y_train - y_mean) / y_std
            n = len(X_train)
            
            best_lml = -np.inf
            best_ls = length_scale
            
            ls_candidates = np.logspace(-1, 1, 10)
            
            for ls in ls_candidates:
                K = kernel(X_train, X_train, ls)
                K += noise_var * np.eye(n)
                K += 1e-8 * np.eye(n)
                
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    lml = -0.5 * (Y_norm @ alpha + log_det + n * np.log(2 * np.pi))
                    
                    if lml > best_lml:
                        best_lml = lml
                        best_ls = ls
                except np.linalg.LinAlgError:
                    continue
                    
            return best_ls

        # Initial fit
        L, alpha, y_mean, y_std = fit_gp(X_train, Y_train, length_scale)
        ls = length_scale
        
        def gp_predict_fast(X_new):
            """Predict mean and variance using pre-computed L and alpha."""
            K_new = kernel(X_train, X_new, ls)
            mean_norm = K_new.T @ alpha
            v = np.linalg.solve(L, K_new.T)
            var_norm = signal_var - np.sum(v ** 2, axis=0)
            var_norm = np.maximum(var_norm, 1e-8)
            
            mean = mean_norm * y_std + y_mean
            var = var_norm * (y_std ** 2)
            
            return mean, var

        def expected_improvement(x):
            """Compute EI for a single point x."""
            x = np.array([x])
            mean, var = gp_predict_fast(x)
            sigma = np.sqrt(var[0])
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mean[0]) / sigma
            ei = (f_best - mean[0]) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization

        # Main optimization loop
        calls_made = n_init
        
        for i in range(budget - n_init):
            # Periodically update hyperparameters
            if i > 0 and i % 20 == 0:
                new_ls = update_gp_hyperparams(X_train, Y_train)
                if new_ls != ls:
                    ls = new_ls
                    L, alpha, y_mean, y_std = fit_gp(X_train, Y_train, ls)
            
            # Find next point using EI
            n_restarts = 10
            start_points = [x_best]
            start_points.extend(rng.uniform(-5, 5, dim) for _ in range(n_restarts - 1))
            
            best_ei = -np.inf
            x_next = None
            
            for x0 in start_points:
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(
                        expected_improvement,
                        x0,
                        bounds=[(-5, 5)] * dim,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-6}
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        x_next = np.clip(res.x, -5, 5)
                except Exception:
                    continue
            
            if x_next is None:
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate the new point
            f_new = func(x_next)
            
            # Update training data
            X_train = np.vstack([X_train, x_next])
            Y_train = np.append(Y_train, f_new)
            
            calls_made += 1
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_next.copy()
            
            # Update GP internals for next iteration
            # We can just refit every time for simplicity since N is small, 
            # or update incrementally. Refitting is O(N^3) which is ~125k ops for N=50.
            # 500 iterations * 125k = 62.5M ops. This is fast.
            L, alpha, y_mean, y_std = fit_gp(X_train, Y_train, ls)
        
        return x_best, f_best
# EVOLVE-BLOCK-END
