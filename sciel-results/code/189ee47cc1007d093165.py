import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Gaussian Process for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Initial random sampling to build GP prior
        n_init = 30
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(np.clip(x, -5, 5)) for x in X])
        calls_made += n_init

        best_idx = np.argmin(Y)
        best_f = Y[best_idx]
        best_x = X[best_idx].copy()

        # Phase 2: GP-based Bayesian Optimization
        # Matern 5/2 kernel with automatic relevance determination
        def matern52_kernel(X1, X2, length_scales, variance):
            """Compute Matern 5/2 kernel matrix between X1 and X2."""
            # Standardize by length scales
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            d = np.sqrt(np.sum((diff / length_scales[np.newaxis, np.newaxis, :]) ** 2, axis=-1))
            sqrt3_d = np.sqrt(3) * d
            K = (1 + sqrt3_d + (sqrt3_d ** 2) / 3) * np.exp(-sqrt3_d) * variance
            return K

        def compute_gp_posterior(X_train, Y_train, X_test, length_scales, variance, noise):
            """Compute GP posterior mean and variance at X_test given training data."""
            n = len(X_train)
            K_train = matern52_kernel(X_train, X_train, length_scales, variance)
            K_train += noise * np.eye(n)
            K_test = matern52_kernel(X_test, X_train, length_scales, variance)
            
            try:
                L = np.linalg.cholesky(K_train + 1e-10 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                try:
                    K_inv = np.linalg.pinv(K_train + 1e-10 * np.eye(n))
                    alpha = K_inv @ Y_train
                    L = None
                except:
                    return np.zeros(len(X_test)), np.ones(len(X_test)) * variance

            if L is not None:
                v = np.linalg.solve(L, K_test.T)
            else:
                v = K_test.T @ np.linalg.pinv(K_train + 1e-10 * np.eye(n))

            mu = K_test @ alpha
            v2 = np.sum(v ** 2, axis=0)
            var = variance - v2
            var = np.maximum(var, 1e-10)  # Ensure positive variance

            return mu, var

        # GP hyperparameters (fixed for simplicity and speed)
        length_scales = np.ones(dim) * 1.0
        variance = 1.0
        noise = 1e-6

        # Main BO loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # If we have enough data, use GP; otherwise, random sampling
            if len(X) < 10:
                # Continue random sampling
                x_new = rng.uniform(-5, 5, size=(1, dim))
                f_new = func(np.clip(x_new[0], -5, 5))
                calls_made += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new[0].copy()
                continue

            # Compute GP posterior on a candidate set
            # Use a mix of random points and points near the current best
            n_candidates = 100
            X_rand = rng.uniform(-5, 5, size=(n_candidates, dim))
            # Also add points in a local neighborhood of the best
            local_radius = 1.0
            X_local = best_x + rng.normal(0, local_radius, size=(n_candidates, dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand = np.vstack([X_rand, X_local])

            mu, var = compute_gp_posterior(X, Y, X_cand, length_scales, variance, noise)

            # Expected Improvement acquisition function
            f_best = best_f
            sigma = np.sqrt(var)
            z = (f_best - mu) / sigma
            EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

            # Select the point with highest EI
            best_ei_idx = np.argmax(EI)
            x_next = X_cand[best_ei_idx]
            x_next = np.clip(x_next, -5, 5)

            f_next = func(x_next)
            calls_made += 1

            # Update training data
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)

            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()

            # Periodically adapt hyperparameters based on recent data
            if calls_made % 50 == 0 and len(X) > 20:
                # Simple adaptation: estimate length scales from data spread
                recent_X = X[-20:]
                recent_Y = Y[-20:]
                # Estimate length scales as typical distance between points with similar Y values
                # This is a crude heuristic; in practice, one would optimize hyperparameters
                pass

        return best_f
