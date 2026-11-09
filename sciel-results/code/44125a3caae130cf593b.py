"""Surrogate-assisted optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Gaussian Process surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _gp_predict(self, X, X_train, y_train):
        """
        Predict mean and variance using a simple GP with RBF kernel.
        To avoid full matrix inversion for every prediction, we use a
        simplified approach: for small n, direct computation is fine.
        """
        n = len(X_train)
        if n == 0:
            return np.zeros(len(X)), np.ones(len(X))

        # Normalize training data to [-1, 1] for better conditioning
        x_min = X_train.min(axis=0)
        x_max = X_train.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-10] = 1.0
        X_train_norm = 2.0 * (X_train - x_min) / x_range - 1.0
        X_norm = 2.0 * (X - x_min) / x_range - 1.0

        # RBF kernel with length-scale per dimension (isotropic for simplicity)
        # Use a moderate length-scale
        l = 0.5
        sigma_f = 1.0
        sigma_n = 1e-6

        # Compute kernel matrix K
        # K[i,j] = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / l^2)
        diff = X_train_norm[:, np.newaxis, :] - X_train_norm[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = sigma_f ** 2 * np.exp(-0.5 * sq_dist / (l ** 2))
        K[np.diag_indices(n)] += sigma_n ** 2

        # Solve for weights
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            # Fallback: add jitter
            K[np.diag_indices(n)] += 1e-4
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                # Last resort: use simple average
                return np.full(len(X), y_train.mean()), np.full(len(X), y_train.var() + 1.0)

        # Predict at X
        diff_test = X_norm[:, np.newaxis, :] - X_train_norm[np.newaxis, :, :]
        sq_dist_test = np.sum(diff_test ** 2, axis=-1)
        K_star = sigma_f ** 2 * np.exp(-0.5 * sq_dist_test / (l ** 2))

        mu = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = sigma_f ** 2 - np.sum(v ** 2, axis=1)
        var = np.maximum(var, 1e-8)

        return mu, np.sqrt(var)

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial points: center + a few random points
        X_list = []
        y_list = []

        # Start with center
        x0 = np.zeros(dim)
        y0 = func(x0)
        X_list.append(x0)
        y_list.append(y0)

        # Add a few random initial points for better landscape coverage
        for _ in range(min(5, budget - 1)):
            x = rng.uniform(-5, 5, dim)
            y = func(x)
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list)
        y = np.array(y_list)

        best_x = X[np.argmin(y)]
        best_y = min(y)

        calls_made = len(X_list)

        # Main loop: surrogate-assisted search
        while calls_made < budget:
            # Fit GP to current data
            mu, sigma = self._gp_predict(np.array([best_x]), X, y)

            # Generate candidate points using a mix of:
            # 1. Local search around best point
            # 2. Global exploration
            n_candidates = min(20, budget - calls_made)

            candidates = []

            # Local candidates around best point
            for _ in range(n_candidates // 2):
                step = 0.5 * rng.standard_normal(dim)
                x_cand = np.clip(best_x + step, -5, 5)
                candidates.append(x_cand)

            # Global candidates
            for _ in range(n_candidates - n_candidates // 2):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)

            candidates = np.array(candidates)

            # Predict mean and variance for candidates
            mu_cand, sigma_cand = self._gp_predict(candidates, X, y)

            # Expected Improvement acquisition
            # EI = (mu_best - mu) * Phi((mu_best - mu)/sigma) + sigma * phi((mu_best - mu)/sigma)
            # where mu_best is the current best (minimum)
            z = (best_y - mu_cand) / (sigma_cand + 1e-10)
            ei = (best_y - mu_cand) * norm.cdf(z) + sigma_cand * norm.pdf(z)

            # Select candidate with highest EI
            idx = np.argmax(ei)
            x_next = candidates[idx]
            y_next = func(x_next)
            calls_made += 1

            # Update data
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

            # Update best
            if y_next < best_y:
                best_y = y_next
                best_x = x_next

        return best_x, best_y
# EVOLVE-BLOCK-END
