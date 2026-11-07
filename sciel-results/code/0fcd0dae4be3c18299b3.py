"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with GP surrogate for low-dimensional search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls_used
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # GP surrogate using RBF kernel
        class GPSurrogate:
            def __init__(self, dim, length_scale=1.0, variance=1.0, noise=1e-8):
                self.dim = dim
                self.length_scale = length_scale
                self.variance = variance
                self.noise = noise
                self.X = None
                self.Y = None

            def rbf_kernel(self, X1, X2):
                # Squared exponential kernel
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=-1)
                return self.variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

            def fit(self, X, Y):
                self.X = X.copy()
                self.Y = Y.copy()
                n = len(X)
                K = self.rbf_kernel(X, X) + self.noise * np.eye(n)
                try:
                    self.L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
                except np.linalg.LinAlgError:
                    # Fallback if not positive definite
                    K += 1e-6 * np.eye(n)
                    self.L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))

            def predict(self, X_new):
                n = len(self.X)
                K_s = self.rbf_kernel(X_new, self.X)
                K_ss = self.rbf_kernel(X_new, X_new)

                # Mean prediction
                mean = K_s @ self.alpha

                # Variance prediction
                v = np.linalg.solve(self.L, K_s.T)
                var = np.diag(K_ss) - np.sum(v ** 2, axis=1)
                var = np.maximum(var, 1e-10)

                return mean, var

            def expected_improvement(self, X_new, f_best, xi=0.01):
                mean, var = self.predict(X_new)
                std = np.sqrt(var)
                z = (f_best - mean - xi) / std
                ei = (f_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
                ei = np.maximum(ei, 0)
                return ei

        # Phase 1: Initial sampling to build GP model
        n_init = 20
        X_samples = np.zeros((n_init, dim))
        Y_samples = np.zeros(n_init)

        # Include center and some spread
        X_samples[0] = np.zeros(dim)
        for i in range(1, n_init):
            X_samples[i] = rng.uniform(-4, 4, size=dim)

        for i in range(n_init):
            if calls_used >= budget:
                break
            if time.time() > deadline - 30.0:
                break
            x = np.clip(X_samples[i], -5, 5)
            f_val = func(x)
            calls_used += 1
            Y_samples[i] = f_val
            update_best(x, f_val)

        if calls_used >= budget or time.time() > deadline - 30.0:
            return best_x, best_f

        # Phase 2: Bayesian Optimization loop
        gp = GPSurrogate(dim, length_scale=1.5, variance=4.0, noise=1e-6)
        gp.fit(X_samples[:n_init], Y_samples[:n_init])

        n_bo = min(300, budget - calls_used)
        candidate_pool_size = 2000

        for step in range(n_bo):
            if calls_used >= budget:
                break
            if time.time() > deadline - 20.0:
                break

            # Generate candidate points
            candidates = rng.uniform(-5, 5, size=(candidate_pool_size, dim))

            # Compute EI for all candidates
            f_best = best_f
            ei = gp.expected_improvement(candidates, f_best, xi=0.05)

            # Select best candidate by EI
            best_idx = np.argmax(ei)
            x_next = np.clip(candidates[best_idx], -5, 5)

            # Evaluate
            f_next = func(x_next)
            calls_used += 1
            update_best(x_next, f_next)

            # Update GP model
            X_samples = np.vstack([X_samples, x_next])
            Y_samples = np.append(Y_samples, f_next)

            # Refit GP periodically (every 10 iterations to save time)
            if (step + 1) % 10 == 0 or step == n_bo - 1:
                gp.fit(X_samples, Y_samples)

        # Phase 3: (1+1)-ES polish from best point found
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.2

            polish_calls = 0
            max_polish = min(50, budget - calls_used)
            for _ in range(max_polish):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.03
                    update_best(x, fx)
                else:
                    sigma *= 0.85

            # Fine refinement
            sigma = max(sigma, 0.05)
            max_fine = min(30, budget - calls_used)
            for _ in range(max_fine):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.01
                    update_best(x, fx)
                else:
                    sigma *= 0.92

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
