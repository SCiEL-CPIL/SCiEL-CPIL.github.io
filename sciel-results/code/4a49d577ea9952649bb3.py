"""Bayesian Optimization with Gaussian Process surrogate for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Initialize with a small set of random points
        n_init = min(10, budget)
        X = np.random.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])
        best_val = np.min(Y)
        best_x = X[np.argmin(Y)]

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6

        # Acquisition: Expected Improvement with small exploration constant
        xi = 1e-3

        calls_made = n_init
        remaining = budget - n_init

        # We'll use a simple GP with RBF kernel
        # To keep it fast, we'll use a limited number of iterations and
        # only refit the GP when we have enough points

        # Pre-allocate for efficiency
        for i in range(remaining):
            if time.time() > deadline:
                break

            # If we have enough points, fit GP and use acquisition
            if len(X) >= 5:
                # Standardize inputs for numerical stability
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0)
                X_std[X_std < 1e-8] = 1.0
                X_norm = (X - X_mean) / X_std

                # Compute kernel matrix
                K = self._rbf_kernel(X_norm, X_norm, length_scale)
                K += noise_var * np.eye(len(X))

                # Standardize outputs
                y_mean = Y.mean()
                y_std = Y.std()
                if y_std < 1e-8:
                    y_std = 1.0
                y_norm = (Y - y_mean) / y_std

                # Solve for alpha
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                except np.linalg.LinAlgError:
                    # Fallback: use random exploration
                    x_new = np.random.uniform(-5, 5, dim)
                    y_new = func(x_new)
                    calls_made += 1
                    X = np.vstack([X, x_new])
                    Y = np.append(Y, y_new)
                    if y_new < best_val:
                        best_val = y_new
                        best_x = x_new
                    continue

                # Sample candidate points
                n_candidates = 200
                X_cand = np.random.uniform(-5, 5, size=(n_candidates, dim))
                X_cand_norm = (X_cand - X_mean) / X_std

                # Compute kernel between candidates and training points
                K_s = self._rbf_kernel(X_cand_norm, X_norm, length_scale)

                # Predict mean and variance
                mu = K_s @ alpha
                v = np.linalg.solve(L, K_s.T)
                var = signal_var - np.sum(v * K_s, axis=1)
                var = np.maximum(var, 1e-8)

                # Convert back to original scale
                mu_orig = mu * y_std + y_mean
                std_orig = np.sqrt(var) * y_std

                # Expected Improvement
                z = (best_val - mu_orig - xi) / std_orig
                from scipy.stats import norm
                ei = (best_val - mu_orig - xi) * norm.cdf(z) + std_orig * norm.pdf(z)
                ei[std_orig < 1e-8] = 0.0

                # Select best candidate
                idx = np.argmax(ei)
                x_new = X_cand[idx]
            else:
                # Not enough points, use random
                x_new = np.random.uniform(-5, 5, dim)

            # Evaluate
            y_new = func(x_new)
            calls_made += 1

            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_val:
                best_val = y_new
                best_x = x_new

            # Occasionally adapt length scale based on improvement
            if i % 50 == 0 and len(X) > 10:
                # Simple adaptation: if we've been improving, reduce length scale
                # If stagnant, increase it
                recent = Y[-10:]
                if np.min(recent) < best_val * 0.9 or (np.max(recent) - np.min(recent)) < 1e-6:
                    length_scale *= 0.9
                else:
                    length_scale *= 1.1
                length_scale = np.clip(length_scale, 0.1, 5.0)

        return best_x, best_val

    def _rbf_kernel(self, A, B, length_scale):
        """Compute RBF kernel between A and B."""
        # A: (n, d), B: (m, d)
        sq_dist = (
            np.sum(A ** 2, axis=1)[:, None]
            + np.sum(B ** 2, axis=1)[None, :]
            - 2 * A @ B.T
        )
        sq_dist = np.maximum(sq_dist, 0)
        return np.exp(-0.5 * sq_dist / (length_scale ** 2))
# EVOLVE-BLOCK-END
