"""Rewritten optimizer: multi-restart CMA-ES with adaptive budget allocation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        remaining_budget = self.budget
        best_x = None
        best_f = np.inf

        # Phase 1: Quick random sampling to get a good starting point
        n_init = min(20, remaining_budget)
        for i in range(n_init):
            if time.time() > deadline - 5:
                break
            x = np.random.uniform(-4, 4, self.dim)
            f = func(x)
            remaining_budget -= 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES from the best point found
        if remaining_budget > 50 and best_x is not None:
            best_x, best_f = self._cma_es(
                func, best_x, best_f, remaining_budget, deadline
            )
            remaining_budget = self.budget - (self.budget - remaining_budget) - (n_init)
            # Recalculate remaining budget properly
            remaining_budget = self.budget - n_init - (self.budget - remaining_budget)

        # Phase 3: If budget remains, do local refinement with small steps
        if remaining_budget > 10 and best_x is not None and time.time() < deadline - 2:
            best_x, best_f = self._local_refine(
                func, best_x, best_f, remaining_budget, deadline
            )

        # Final check: make sure we don't exceed budget
        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, deadline):
        """CMA-ES implementation."""
        dim = self.dim
        x = x0.copy()
        f = f0
        sigma = 1.0  # initial step size

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - 1 / mu_eff + 2 * min(1, (mu_eff - 1) / (dim + 1))) / (
            (dim + 2) ** 2 + mu_eff
        )
        c_mu = min(
            1 - c_1,
            2 * (
                (sqrt_mu_eff - 2) / (dim + 2) - 1
            ),
        )
        sqrt_mu_eff = np.sqrt(mu_eff)
        c_mu = min(1 - c_1, 2 * ((sqrt_mu_eff - 2) / (dim + 2) - 1))
        c_mu = max(0.0, c_mu)

        # Initialize CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        evals_used = 0
        for gen in range(budget // lambda_pop + 1):
            if evals_used >= budget or time.time() > deadline - 5:
                break

            # Generate population
            offspring = []
            for i in range(lambda_pop):
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = np.array([func(y) for y in offspring])
            evals_used += lambda_pop

            # Sort by fitness
            order = np.argsort(offspring_f)
            y_sorted = np.array([offspring[i] for i in order])
            f_sorted = offspring_f[order]

            # Update best
            if f_sorted[0] < f:
                f = f_sorted[0]
                x = y_sorted[0].copy()

            # Update evolution paths
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                B @ (D * (y_w / sigma))
            )
            h_sig = (
                np.linalg.norm(ps)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
                / chi_n
                < 1.4 + 2 / (dim + 1)
            )
            pc = (1 - c_c) * pc + h_sig * c_c * y_w / sigma

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            outer = np.outer(pc, pc)
            C = (
                (1 + c_1 * delta_h - c_1 - c_mu) * C
                + c_1 * (outer - C)
                + c_mu * np.sum(
                    weights[:, None, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x)
                    / sigma ** 2,
                    axis=0,
                )
            )

            # Eigendecomposition
            C = 0.5 * (C + C.T)  # symmetrize
            try:
                D2, B = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
            except np.linalg.LinAlgError:
                D2 = np.ones(dim)
                B = np.eye(dim)

            # Update sigma
            sigma *= np.exp(
                c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x + np.sum(weights[:, None] * y_sorted[:mu], axis=0) - x
            x = np.clip(x, -5, 5)

        return x, f

    def _local_refine(self, func, x0, f0, budget, deadline):
        """Local refinement with decreasing step sizes."""
        dim = self.dim
        x = x0.copy()
        f = f0
        sigma = 0.5

        for i in range(budget):
            if time.time() > deadline - 2:
                break
            for _ in range(3):
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
            sigma *= 0.95

        return x, f
# EVOLVE-BLOCK-END
