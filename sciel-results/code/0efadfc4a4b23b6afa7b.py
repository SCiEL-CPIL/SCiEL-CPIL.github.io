"""Optimized Bayesian Optimization for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a small random sample
        n_init = 10
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])

        # GP hyperparameters
        # Weierstrass has oscillations at multiple scales
        # Start with moderate length scales
        lengthscales = np.ones(dim) * 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def compute_kernel(X1, X2, ls):
            """Compute Matern-5/2 kernel between X1 and X2."""
            # X1: (n1, dim), X2: (n2, dim)
            # ls: scalar or (dim,)
            if np.isscalar(ls):
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist) / ls
            else:
                # Anisotropic
                # Scale each dimension by 1/ls
                scaled_X1 = X1 / ls
                scaled_X2 = X2 / ls
                diff = scaled_X1[:, np.newaxis, :] - scaled_X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist)
            
            return signal_var * (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)

        X_list = X.copy()
        Y_list = Y.copy()

        # Pre-compute grid points for acquisition function maximization
        # Use a grid of 4 points per dimension: 4^5 = 1024
        grid_vals = np.linspace(-5, 5, 4)
        grid = np.array(np.meshgrid(*[grid_vals]*dim)).T.reshape(-1, dim)

        for i in range(budget - n_init):
            # Update GP model
            n = len(Y_list)
            K = compute_kernel(X_list, X_list, lengthscales)
            K += noise_var * np.eye(n)
            
            # Cholesky
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
            
            # Pre-compute alpha = K^{-1} Y
            v = np.linalg.solve(L, Y_list)
            alpha = np.linalg.solve(L.T, v)
            
            # Evaluate acquisition function on grid
            k_grid = compute_kernel(grid, X_list, lengthscales)
            
            # mu: (n_grid,)
            mu = k_grid @ alpha
            
            # var: (n_grid,)
            v_grid = np.linalg.solve(L, k_grid.T)
            var = signal_var - np.sum(v_grid * v_grid, axis=0)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            
            # Expected Improvement
            y_best = np.min(Y_list)
            xi = 0.01
            z = (mu - y_best - xi) / sigma
            ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best point from grid
            idx = np.argmax(ei)
            x_new = grid[idx].copy()
            
            # Refine with L-BFGS-B on the acquisition function
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                k_x = compute_kernel(x.reshape(1, dim), X_list, lengthscales)[0]
                mu_x = k_x @ alpha
                v_x = np.linalg.solve(L, k_x)
                var_x = signal_var - np.sum(v_x * v_x)
                var_x = max(var_x, 1e-10)
                sigma_x = np.sqrt(var_x)
                z_x = (mu_x - y_best - xi) / sigma_x
                ei_x = (mu_x - y_best - xi) * norm.cdf(z_x) + sigma_x * norm.pdf(z_x)
                return -ei_x
            
            # Try refinement from the best grid point
            try:
                res = minimize(neg_ei, x_new, method='L-BFGS-B', 
                               bounds=[(-5, 5)]*dim, options={'maxiter': 50})
                if res.fun < neg_ei(x_new):
                    x_new = np.clip(res.x, -5, 5)
            except Exception:
                pass
            
            # Evaluate
            y_new = func(x_new)
            
            # Append
            X_list = np.vstack([X_list, x_new])
            Y_list = np.append(Y_list, y_new)
            
            # Periodically adapt length scale based on observed scale
            if i % 50 == 0 and i > 0:
                # Simple adaptation: if we're making progress, keep; otherwise adjust
                # For now, keep fixed for stability
                pass

        return X_list[-1], Y_list[-1]
# EVOLVE-BLOCK-END
