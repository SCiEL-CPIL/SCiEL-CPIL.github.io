"""Surrogate-based optimizer using Gaussian Process for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-based optimizer: GP model + Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling - mix of random and structured points
        n_init = min(40, budget)
        X = np.zeros((n_init, dim))
        fvals = np.zeros(n_init)

        # First point: center
        X[0] = np.zeros(dim)
        fvals[0] = func(X[0])

        # Second point: random
        X[1] = rng.uniform(-5, 5, dim)
        fvals[1] = func(X[1])

        # Remaining initial points: mix of random and grid-like
        for i in range(2, n_init):
            if (time.time() - start_time) > deadline:
                break
            # Alternate between random and structured exploration
            if i % 3 == 0:
                # Structured: random point near center
                X[i] = rng.uniform(-2, 2, dim)
            elif i % 3 == 1:
                # Random in full space
                X[i] = rng.uniform(-5, 5, dim)
            else:
                # Random but slightly biased toward center
                X[i] = rng.normal(0, 2.5, dim)
                X[i] = np.clip(X[i], -5, 5)
            fvals[i] = func(X[i])

        # Track best
        best_idx = np.argmin(fvals)
        best_x = X[best_idx].copy()
        best_f = fvals[best_idx]

        # Phase 2: Surrogate-based optimization
        # Use a simple RBF surrogate (faster than GP for small datasets)
        n_current = n_init
        calls_made = n_init

        # RBF kernel parameters
        length_scale = 1.5
        noise_var = 1e-6

        def rbf_kernel(X1, X2, ls):
            """Compute RBF kernel matrix between X1 and X2."""
            # X1: (n1, d), X2: (n2, d)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            return np.exp(-sq_dist / (2.0 * ls ** 2))

        def predict_rbf(X_new, X_train, y_train, ls, noise):
            """Predict mean and variance using RBF interpolation."""
            K = rbf_kernel(X_train, X_train, ls) + noise * np.eye(len(X_train))
            K_new = rbf_kernel(X_new, X_train, ls)

            # Solve for weights
            try:
                alpha = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]

            mean = K_new @ alpha
            # Variance: diagonal of K_new @ K^{-1} @ K_new.T - K_new @ K^{-1} @ K_new.T
            # Simpler: var = 1 - K_new @ K^{-1} @ K_new.T (for standardized)
            K_inv = np.linalg.pinv(K)
            var = 1.0 - np.sum(K_new @ K_inv * K_new, axis=1)
            var = np.maximum(var, 0.0)
            return mean, var

        def expected_improvement(mean, std, best_f, xi=0.01):
            """Compute EI acquisition function."""
            std = np.maximum(std, 1e-8)
            improvement = best_f - mean - xi
            z = improvement / std
            # Normal CDF and PDF
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei = np.where(improvement > 0, ei, 0.0)
            return ei

        # Adaptive length scale
        ls = length_scale

        for iteration in range(budget - calls_made):
            if (time.time() - start_time) > deadline:
                break

            # If we have enough points, use surrogate
            if n_current >= 5:
                # Predict on a candidate set
                n_candidates = 200
                # Generate candidates: mix of random and perturbations of best
                candidates = np.zeros((n_candidates, dim))
                n_perturb = n_candidates // 3
                n_random = n_candidates // 3
                n_near_best = n_candidates - n_perturb - n_random

                # Perturbations around best
                candidates[:n_perturb] = best_x + rng.normal(0, 1.0, (n_perturb, dim))
                candidates[:n_perturb] = np.clip(candidates[:n_perturb], -5, 5)

                # Random points
                candidates[n_perturb:n_perturb + n_random] = rng.uniform(-5, 5, (n_random, dim))

                # Points near current best with larger step
                candidates[n_perturb + n_random:] = best_x + rng.normal(0, 2.0, (n_near_best, dim))
                candidates[n_perturb + n_random:] = np.clip(candidates[n_perturb + n_random:], -5, 5)

                # Predict
                mean, var = predict_rbf(candidates, X[:n_current], fvals[:n_current], ls, noise_var)
                std = np.sqrt(var)

                # Compute EI
                ei = expected_improvement(mean, std, best_f, xi=0.01)

                # Select best candidate
                best_cand_idx = np.argmax(ei)
                x_new = candidates[best_cand_idx]
            else:
                # Not enough data yet, just sample randomly
                x_new = rng.uniform(-5, 5, dim)

            f_new = func(x_new)
            calls_made += 1

            # Update data
            if n_current < budget:
                X[n_current] = x_new
                fvals[n_current] = f_new
                n_current += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                # Reset length scale on improvement (exploration)
                ls = 1.5
            else:
                # Gradually decrease length scale (exploitation)
                ls = max(0.3, ls * 0.99)

            # Check time periodically
            if iteration % 10 == 0 and (time.time() - start_time) > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
