"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and initial exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0

        def _time_ok():
            return time.time() < deadline

        # Initial broad sampling
        n_initial = 40
        candidates = []
        for _ in range(n_initial):
            if not _time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))
            calls_used += 1

        candidates.sort(key=lambda t: t[0])
        best_f = candidates[0][0]
        best_x = candidates[0][1].copy()

        # Use top candidates as starting points for multi-start CMA-ES
        # Strategy: run several CMA-ES instances from different starting points
        # with decreasing budget allocation

        # CMA-ES parameters
        def _cmaes(start_x, start_f, n_steps, sigma0, rng):
            """Run CMA-ES for n_steps from start_x."""
            nonlocal calls_used, best_f, best_x

            n = dim
            lam = 4 + int(3 * np.log(n))  # population size
            mu = lam // 2  # number of best solutions used for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            # Adaptation parameters
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / (n**1.7 + 10 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_mu
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            # State
            mean = start_x.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            local_best_f = start_f
            local_best_x = start_x.copy()

            for step in range(n_steps):
                if not _time_ok() or calls_used >= budget:
                    break

                # Generate offspring
                population = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x_i = np.clip(mean + sigma * y, -5, 5)
                    f_i = func(x_i)
                    calls_used += 1
                    population.append((f_i, x_i, y, z))

                population.sort(key=lambda t: t[0])

                # Update best
                if population[0][0] < local_best_f:
                    local_best_f = population[0][0]
                    local_best_x = population[0][1].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * population[i][2]

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / chi_n * (B.T @ (D * y_w))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + c_c * np.sign(local_best_f < local_best_f) * (B.T @ (D * y_w))

                # Rank-one and rank-mu update
                C = (1 + c_1 * c_c * 2 - c_1 - c_mu * c_c) * C \
                    + c_1 * (p_c[:, None] * p_c[None, :]) \
                    + c_mu * np.sum(weights[i][:, None, None] * y_w_i[:, None] * y_w_i[None, :]
                                     for i, y_w_i in [
                                         (population[i][2], population[i][2]) for i in range(mu)
                                     ])

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update B and D
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update mean
                mean = np.clip(mean + sigma * (B @ (D * y_w)), -5, 5)

                # Check stagnation: if no improvement for many steps, break
                if local_best_f >= start_f - 1e-10 and step > 20:
                    # Some progress check
                    pass

                # Adaptive stopping: if sigma is very small or step is too large
                if sigma < 1e-8 or sigma > 100:
                    break

            return local_best_x, local_best_f

        # Multi-start strategy
        # We'll run CMA-ES from a few of the best initial points
        n_starts = min(3, len(candidates))
        starts = [candidates[i][1] for i in range(n_starts)]
        start_fvals = [candidates[i][0] for i in range(n_starts)]

        # Distribute remaining budget
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # Each start gets a portion of the budget
        # First start gets more budget
        budgets_per_start = []
        for i in range(n_starts):
            if i == 0:
                b = int(remaining * 0.5)
            else:
                b = int(remaining * 0.25)
            budgets_per_start.append(b)

        # Adjust to not exceed remaining
        total = sum(budgets_per_start)
        if total > remaining:
            scale = remaining / total
            budgets_per_start = [int(b * scale) for b in budgets_per_start]

        for i in range(n_starts):
            if not _time_ok() or calls_used >= budget:
                break

            b = budgets_per_start[i]
            if b < 10:
                continue

            # Use the best found so far as the starting point for later starts
            if i > 0:
                sx = best_x.copy()
                sf = best_f
            else:
                sx = starts[i].copy()
                sf = start_fvals[i]

            # CMA-ES sigma: start with a reasonable value
            sigma0 = 2.0 if i == 0 else 1.0

            _, _ = _cmaes(sx, sf, b, sigma0, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
