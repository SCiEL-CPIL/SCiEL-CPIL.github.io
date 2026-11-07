"""Optimized optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and stagnation detection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_step(self, x, C, sigma, func, rng, max_iter, deadline):
        """
        Run a simplified CMA-ES for max_iter evaluations.
        Returns (x_best, f_best, sigma_final, C_final).
        """
        n = self.dim
        # Initialize weights
        mu = max(1, n // 2)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Step size control
        chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Population
        lambda_pop = 2 * n
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        d_orth = 1.0

        x_best = x.copy()
        f_best = func(x)

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                ys[i] = x + sigma * C @ rng.standard_normal(n)

            # Evaluate
            fs = np.array([func(np.clip(y, -5, 5)) for y in ys])

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update best
            if fs[0] < f_best:
                f_best = fs[0]
                x_best = ys[0].copy()

            # Mean update
            y_mean = np.average(ys[:mu], axis=0, weights=weights)
            x_new = x + y_mean

            # Update C
            delta_sigma = np.linalg.norm(y_mean) / chi_n - 1.0
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_sigma * (y_mean / max(np.linalg.norm(y_mean), 1e-16))
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chi_n < 1.4 + 2 / (n + 1)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            v = p_c @ C @ p_c
            C = (1 + c_1 * (2 - h_sigma) + c_mu * (h_sigma - 1)) * C \
                - c_1 * C @ C \
                + c_1 * (1 + (2 - h_sigma) * (1 - c_c) / c_c - 1) * np.outer(p_c, p_c) \
                + c_mu / mu_eff * sum(w * np.outer(ys[i] - x, ys[i] - x) for i, w in enumerate(weights))

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            x = np.clip(x_new, -5, 5)

            # Stagnation check
            if it > 10 and np.all(np.abs(ys[:mu] - x) < 1e-6):
                break

        return x_best, f_best, sigma, C

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim

        # Phase 1: Multi-start to find promising basins
        # Use 8 restarts with varying initializations
        n_starts = 8
        budget_per_start = max(10, self.budget // n_starts - 5)

        best_x = None
        best_f = np.inf
        best_sigma = 1.0
        best_C = None

        for s in range(n_starts):
            if time.time() > deadline - 10:
                break

            # Different initialization strategies
            if s == 0:
                x0 = rng.uniform(-5, 5, n)
            elif s == 1:
                x0 = np.zeros(n)
            elif s == 2:
                x0 = rng.uniform(-2, 2, n)
            elif s == 3:
                x0 = rng.uniform(-4, 4, n)
            elif s == 4:
                x0 = np.sign(rng.standard_normal(n)) * rng.uniform(1, 4, n)
            elif s == 5:
                x0 = rng.uniform(-1, 1, n)
            elif s == 6:
                x0 = rng.uniform(-3, 3, n)
            else:
                x0 = rng.uniform(-5, 5, n)

            x0 = np.clip(x0, -5, 5)
            C0 = np.eye(n)
            sigma0 = 2.0

            # Run CMA-ES for this start
            x_best, f_best, sigma_end, C_end = self._cma_step(
                x0, C0, sigma0, func, rng, budget_per_start, deadline
            )

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                best_sigma = max(sigma_end, 0.1)
                best_C = C_end.copy()

        # Phase 2: Exploit the best basin found
        if time.time() < deadline - 5 and best_x is not None:
            # Continue with refined CMA-ES from best point
            C_refined = best_C if best_C is not None else np.eye(n)
            sigma_refined = min(best_sigma, 1.0)

            # Run for remaining budget
            remaining = self.budget - n_starts * budget_per_start
            remaining = max(remaining, 20)

            x_final, f_final, _, _ = self._cma_step(
                best_x, C_refined, sigma_refined, func, rng, remaining, deadline
            )

            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
