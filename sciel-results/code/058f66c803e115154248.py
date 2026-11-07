"""Bayesian Optimization for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Storage for observed points and values
        X = np.empty((0, dim))
        Y = np.empty((0,))

        # Initial exploration: random + grid-like points
        n_init = min(30, budget)
        # Use a mix of random and structured points
        init_points = []
        # Random points
        init_points.append(rng.uniform(-5, 5, (n_init // 2 + 1, dim)))
        # Structured: midpoints of sub-boxes
        if n_init >= 10:
            grid_size = 2
            coords = np.linspace(-4, 4, grid_size)
            grid = np.array(np.meshgrid(*[coords] * dim), dtype=float)
            grid = grid.reshape(-1, dim)
            # Sample from grid
            n_grid = min(n_init // 2, len(grid))
            if n_grid > 0:
                idx = rng.choice(len(grid), size=n_grid, replace=False)
                init_points.append(grid[idx])
        # Add center
        init_points.append(np.zeros((1, dim)))

        X_init = np.vstack(init_points)
        # Clip to box
        X_init = np.clip(X_init, -5, 5)
        # Remove duplicates approximately
        X_init = self._dedupe(X_init)

        for i in range(min(len(X_init), budget)):
            x = X_init[i]
            fx = func(x)
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, fx)
            if time.time() > deadline:
                return X[np.argmin(Y)], Y.min()

        # Bayesian optimization loop
        calls_made = len(Y)
        best_fx = Y.min()

        # GP hyperparameters
        # We'll use a simple Matern kernel with ARD
        # For speed, we'll use a simplified GP with fixed hyperparameters
        # or a very simple one

        # Actually, for 500 calls in 5D, full GP might be slow.
        # Let's use a lightweight approach: 
        # 1. Use a simple radial basis kernel with fixed length scales
        # 2. Or use a simpler surrogate like Kriging with fixed params
        
        # Let's try a simple GP with fixed hyperparameters for speed
        # Length scales: start with 2.0, noise: 1e-4
        
        length_scales = np.ones(dim) * 2.0
        noise_var = 1e-4

        def gp_predict(X_train, Y_train, X_test):
            """Predict mean and variance using GP with Matern-5/2 kernel."""
            # Normalize inputs for better conditioning
            # But we'll work in original space with length scales
            
            n = len(X_train)
            d = len(X_test)
            
            # Compute kernel matrix K(X_train, X_train)
            K = self._matern_kernel(X_train, X_train, length_scales) + noise_var * np.eye(n)
            K += 1e-6 * np.eye(n)  # Jitter for numerical stability
            
            # Compute K(X_train, X_test)
            Ks = self._matern_kernel(X_train, X_test, length_scales)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, Ks)
                Kss = self._matern_kernel(X_test, X_test, length_scales)
                mean = Ks.T @ alpha
                var = np.diag(Kss) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
                mean = Ks.T @ alpha
                var = np.ones(d) * Y_train.var() + 1e-10
            
            return mean, var

        def _matern_kernel(self, X1, X2, ls):
            """Matern-5/2 kernel."""
            # X1: (n, d), X2: (m, d)
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            # Divide by length scales
            diff_scaled = diff / ls[np.newaxis, np.newaxis, :]
            r = np.sqrt(np.sum(diff_scaled**2, axis=2))
            # Matern-5/2: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
            sqrt3 = np.sqrt(3)
            K = (1 + sqrt3 * r) * np.exp(-sqrt3 * r)
            return K

        def expected_improvement(mean, var, best_f, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            imp = best_f - mean - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = imp / sigma
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(sigma > 1e-8, ei, 0)
            return ei

        # Main BO loop
        for i in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            # If we have few points, do random exploration
            if len(Y) < 10:
                x_new = rng.uniform(-5, 5, dim)
            else:
                # Evaluate EI on a candidate set
                # Generate candidates: random + local perturbations around best
                n_cand = 500
                candidates = rng.uniform(-5, 5, (n_cand, dim))
                
                # Add local candidates around current best
                x_best = X[np.argmin(Y)]
                local_cands = x_best + rng.normal(0, 0.5, (n_cand // 2, dim))
                local_cands = np.clip(local_cands, -5, 5)
                candidates = np.vstack([candidates, local_cands])
                
                # Predict
                mean, var = gp_predict(X, Y, candidates)
                
                # Compute EI
                ei = expected_improvement(mean, var, best_fx)
                
                # Select best candidate
                x_new = candidates[np.argmax(ei)]
            
            x_new = np.clip(x_new, -5, 5)
            fx_new = func(x_new)
            calls_made += 1
            
            X = np.vstack([X, x_new.reshape(1, -1)])
            Y = np.append(Y, fx_new)
            
            if fx_new < best_fx:
                best_fx = fx_new

        return X[np.argmin(Y)], Y.min()

    def _dedupe(self, X, tol=1e-6):
        """Remove approximately duplicate points."""
        if len(X) == 0:
            return X
        unique = [X[0]]
        for x in X[1:]:
            if all(np.linalg.norm(x - u) > tol for u in unique):
                unique.append(x)
        return np.array(unique)
# EVOLVE-BLOCK-END
