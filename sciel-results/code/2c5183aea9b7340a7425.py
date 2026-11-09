"""Bayesian Optimization with Matern-5/2 GP for Weierstrass f16."""
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

        # Initialize with a stratified random sample
        n_init = 15
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])

        # GP hyperparameters
        # Weierstrass f16 has oscillations at multiple scales
        # Start with moderate length scales
        lengthscales = np.ones(dim) * 0.8
        signal_var = 1.0
        noise_var = 1e-6

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
                scaled_X1 = X1 / ls
                scaled_X2 = X2 / ls
                diff = scaled_X1[:, np.newaxis, :] - scaled_X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist)
            
            return signal_var * (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)

        X_list = X.copy()
        Y_list = Y.copy()

        # Pre-compute a coarse grid for acquisition function maximization
        # Use a grid of 6 points per dimension: 6^5 = 7776
        grid_vals = np.linspace(-5, 5, 6)
        grid = np.array(np.meshgrid(*[grid_vals]*dim)).T.reshape(-1, dim)

        # Add random points to the grid to improve exploration
        n_random = 500
        random_pts = rng.uniform(-5, 5, (n_random, dim))
        grid = np.vstack([grid, random_pts])

        for i in range(budget - n_init):
            # Update GP model
            n = len(Y_list)
            K = compute_kernel(X_list, X_list, lengthscales)
            K += noise_var * np.eye(n)
            
            # Cholesky with jitter
            jitter = 1e-8
            max_jitter = 1e-2
            L = None
            while L is None:
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    if jitter >= max_jitter:
                        K += 1e-4 * np.eye(n)
                        L = np.linalg.cholesky(K)
                        break
                    K += jitter * np.eye(n)
                    jitter *= 10
            
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
            xi = 0.005
            z = (mu - y_best - xi) / sigma
            ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best point from grid
            idx = np.argmax(ei)
            x_new = grid[idx].copy()
            
            # Refine with local optimization
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                k_x = compute_kernel(x.reshape(1, -1), X_list, lengthscales)
                mu_x = k_x @ alpha
                v_x = np.linalg.solve(L, k_x.T)
                var_x = signal_var - np.sum(v_x * v_x, axis=0)
                var_x = np.maximum(var_x, 1e-10)
                sigma_x = np.sqrt(var_x)
                z_x = (mu_x - y_best - xi) / sigma_x
                ei_x = (mu_x - y_best - xi) * norm.cdf(z_x) + sigma_x * norm.pdf(z_x)
                return -ei_x[0]
            
            # Try a few random restarts for local optimization
            best_x = x_new
            best_f = neg_ei(x_new)
            for restart in range(3):
                x0 = x_new + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5, 5)
                res = minimize(neg_ei, x0, method='Nelder-Mead', 
                              options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            
            x_new = np.clip(best_x, -5, 5)
            
            # Evaluate
            y_new = func(x_new)
            
            # Append
            X_list = np.vstack([X_list, x_new])
            Y_list = np.append(Y_list, y_new)
            
            # Adapt length scale slightly based on progress
            if i % 25 == 0 and i > 0:
                recent = Y_list[-min(10, len(Y_list)):]
                if len(recent) > 5:
                    improvement = recent[-1] - recent[0]
                    if improvement > 0:  # Getting worse (minimizing)
                        lengthscales *= 0.95
                    else:
                        lengthscales *= 1.05
                    lengthscales = np.clip(lengthscales, 0.3, 2.0)
            
            # Occasionally add new random points to grid for exploration
            if i % 50 == 0:
                new_random = rng.uniform(-5, 5, (100, dim))
                grid = np.vstack([grid, new_random])
                # Keep grid size manageable
                if len(grid) > 10000:
                    grid = grid[:10000]

        return X_list[-1], Y_list[-1]
