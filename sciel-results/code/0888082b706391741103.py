"""Bayesian Optimization with Gaussian Process for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian optimization using Gaussian Process regression with Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # GP hyperparameters
        length_scales = np.ones(n)
        variance = 1.0
        noise = 1e-8

        def rbf_kernel(X1, X2, ls, var):
            """RBF kernel between two sets of points."""
            # X1: (N, n), X2: (M, n)
            # Compute squared distances
            sq_dists = (
                np.sum(X1 ** 2, axis=1)[:, None]
                - 2 * X1 @ X2.T
                + np.sum(X2 ** 2, axis=1)[None, :]
            )
            sq_dists = np.maximum(sq_dists, 0)
            # Scale by length scales
            ls_sq = (ls ** 2)[None, None, :]
            sq_dists_scaled = np.sum(sq_dists[:, :, None] / ls_sq, axis=2)
            return var * np.exp(-0.5 * sq_dists_scaled)

        def predict_gp(X, X_train, y_train, ls, var, noise):
            """
            Predict mean and variance at points X given training data.
            X: (M, n)
            X_train: (N, n)
            y_train: (N,)
            Returns: (mean, var) each of shape (M,)
            """
            K = rbf_kernel(X_train, X_train, ls, var)
            K += noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_train, X, ls, var)
            try:
                alpha = np.linalg.solve(K, y_train)
                mean = K_star.T @ alpha
                v = np.linalg.solve(K, K_star)
                var_pred = var - np.sum(K_star ** 2 * v, axis=0)
                var_pred = np.maximum(var_pred, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: use prior
                mean = np.zeros(len(X))
                var_pred = np.full(len(X), var)
            return mean, var_pred

        # Initial sampling: Latin Hypercube-like spread
        n_init = min(10, budget)
        X_train = np.empty((n_init, n))
        y_train = np.empty(n_init)

        # Generate initial points spread across the domain
        for i in range(n_init):
            x = rng.uniform(-5, 5, size=n)
            X_train[i] = x
            y_train[i] = func(x)
            budget -= 1

        best_x = X_train[np.argmin(y_train)].copy()
        best_f = y_train.min()

        # Adaptive exploration parameter
        xi = 0.0  # Exploit from the start

        calls = 0
        while budget > 0 and time.time() < deadline:
            # Check for time
            if time.time() > deadline - 2.0:
                break

            # Generate candidate points for acquisition
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, size=(n_candidates, n))

            # Predict GP at candidates
            mean, var = predict_gp(X_cand, X_train, y_train, length_scales, variance, noise)
            std = np.sqrt(var)

            # Expected Improvement
            f_best = best_f
            eps = 1e-10
            Z = (f_best - mean) / (std + eps)
            from scipy.stats import norm
            EI = (f_best - mean) * norm.cdf(Z) + std * norm.pdf(Z)
            EI -= xi  # Shift for exploration

            # Select best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]

            # Evaluate
            f_new = func(x_new)
            budget -= 1
            calls += 1

            # Update training set
            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, f_new)

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Adaptively adjust hyperparameters (simple heuristic)
            # If we have many points, slightly increase noise to avoid overfitting
            if len(y_train) > 30:
                noise = 1e-6
            elif len(y_train) > 15:
                noise = 1e-7

            # Occasionally try a random restart if stagnation detected
            # Simple stagnation check: no improvement in last 20 evaluations
            if len(y_train) > 20:
                recent_best = np.min(y_train[-20:])
                if recent_best >= best_f + 1e-6:
                    # No improvement, try a larger random jump
                    if budget > 5:
                        x_restart = rng.uniform(-5, 5, size=n)
                        f_restart = func(x_restart)
                        budget -= 1
                        calls += 1
                        X_train = np.vstack([X_train, x_restart])
                        y_train = np.append(y_train, f_restart)
                        if f_restart < best_f:
                            best_f = f_restart
                            best_x = x_restart.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
