"""Optimizer for BBOB f21 (Gallagher 101 peaks) using GP surrogate + CMA-ES polish."""
import time
import numpy as np
from scipy.stats import norm


class Optimizer:
    """GP surrogate global search followed by CMA-ES local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage for observations
        X = []  # list of points
        Y = []  # list of function values

        def add_point(x, fx):
            X.append(np.asarray(x, dtype=np.float64))
            Y.append(fx)

        # Initialize with a small random sample
        n_init = 10
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            add_point(x, fx)

        calls_used = n_init
        best_x = X[np.argmin(Y)]
        best_y = min(Y)

        # Phase 1: GP-based global exploration
        # We'll use the GP for a portion of the budget, then switch to CMA-ES
        gp_budget = min(200, budget - 50)  # reserve some for polish
        gp_iterations = min(gp_budget - calls_used, 150)

        # GP hyperparameters
        length_scale = np.ones(dim)
        noise = 1e-4

        def matern52_kernel(X1, X2, ls, sigma):
            """Matern 5/2 kernel."""
            # X1: (n1, d), X2: (n2, d)
            d = X1.shape[1]
            # Compute squared distances
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            r = np.sqrt(sq_dist) / ls
            K = (1 + r + r**2 / 3.0) * np.exp(-r)
            return sigma**2 * K

        def gp_predict(X_new, X_train, Y_train, ls, sigma, noise):
            """Predict mean and variance at X_new."""
            n = len(X_train)
            if n == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0]) * sigma**2

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            K = matern52_kernel(X_train, X_train, ls, sigma) + noise * np.eye(n)
            K_star = matern52_kernel(X_new, X_train, ls, sigma)
            K_star_star = matern52_kernel(X_new, X_new, ls, sigma)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback to least squares
                    alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
                    K_inv = np.linalg.pinv(K)
                else:
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

            if 'alpha' not in dir():
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

            mean = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)

            return mean, var

        def expected_improvement(mean, var, best_y, xi=0.01):
            """Compute EI for minimization."""
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            z = (best_y - mean - xi) / std
            ei = (best_y - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Adaptive length scale based on data spread
        def update_length_scale(X_data):
            if len(X_data) < 2:
                return np.ones(dim)
            X_arr = np.array(X_data)
            spread = np.std(X_arr, axis=0)
            spread = np.maximum(spread, 0.5)
            return spread

        for it in range(gp_iterations):
            if time.time() > deadline - 30:
                break
            if calls_used >= budget:
                break

            # Update length scale
            length_scale = update_length_scale(X)

            # Sample candidate points
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, dim))

            # Predict mean and variance
            mean, var = gp_predict(candidates, X, Y, length_scale, 1.0, noise)

            # Compute EI
            ei = expected_improvement(mean, var, best_y, xi=0.01)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]
            fx_new = func(x_new)
            calls_used += 1
            add_point(x_new, fx_new)

            if fx_new < best_y:
                best_y = fx_new
                best_x = x_new.copy()

        # Phase 2: CMA-ES local polish around the best point
        remaining = budget - calls_used
        if remaining > 10:
            # Simple CMA-ES-like local search
            x_curr = best_x.copy()
            f_curr = best_y
            sigma = 0.5  # initial step size for local search

            # Use a simple evolution strategy for local refinement
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break

                # Generate perturbation
                perturbation = rng.standard_normal(dim) * sigma
                x_trial = np.clip(x_curr + perturbation, -5, 5)
                f_trial = func(x_trial)
                calls_used += 1

                if f_trial < f_curr:
                    x_curr = x_trial
                    f_curr = f_trial
                    best_x = x_curr.copy()
                    best_y = f_curr
                    sigma *= 1.2  # increase step size on success
                else:
                    sigma *= 0.85  # decrease step size on failure

                # Occasionally try a larger jump to escape local minima
                if rng.random() < 0.05:
                    x_jump = np.clip(x_curr + rng.standard_normal(dim) * 2.0, -5, 5)
                    f_jump = func(x_jump)
                    calls_used += 1
                    if f_jump < f_curr:
                        x_curr = x_jump
                        f_curr = f_jump
                        best_x = x_curr.copy()
                        best_y = f_curr

        return best_x, best_y
