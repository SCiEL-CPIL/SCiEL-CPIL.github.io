"""CMA-ES optimizer for the BBOB f6 (Attractive sector) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        n = self.dim
        # CMA-ES parameters
        popsize = 4 + int(np.floor(3 * np.log(n)))  # 8 for n=5
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1) ** 2 + c_sigma) / ((n + 1.3) ** 2 + mueff)
        c_mu = (2 * (mueff / (n + 1) ** 2 - c_1)) / (
            (n + 2) ** 2 + mueff / (n + 1) ** 2
        )
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Initial state
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        def inv_chol_update(C, step):
            """Update C = C + step and maintain inverse Cholesky via Sherman-Morrison-ish
            approach. For stability we recompute Cholesky each generation (n=5 is cheap)."""
            return C

        best_x = x.copy()
        best_f = np.inf
        calls = 0
        remaining = self.budget

        # We'll evaluate in batches of popsize
        # First, do a few exploratory evaluations to find a decent starting point
        # Actually, let's just start CMA-ES from origin with sigma=2.0

        while remaining > 0 and time.time() < deadline:
            # Determine how many we can evaluate this generation
            # Must not exceed remaining
            gen_pop = min(popsize, remaining)
            if gen_pop < 1:
                break

            # Generate offspring
            offspring = np.empty((gen_pop, n))
            for i in range(gen_pop):
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt_inv @ z)
                offspring[i] = np.clip(y, -5, 5)
                remaining -= 1
                calls += 1

            # Evaluate
            fvals = np.empty(gen_pop)
            for i in range(gen_pop):
                fvals[i] = func(offspring[i])
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = offspring[i].copy()
                if time.time() > deadline:
                    # Stop evaluating rest of generation, fill with inf
                    for j in range(i + 1, gen_pop):
                        fvals[j] = np.inf
                        # Note: we already decremented remaining for all gen_pop
                        break
                    break

            # Sort by fitness
            order = np.argsort(fvals)
            sorted_offspring = offspring[order]
            sorted_fvals = fvals[order]

            # Only use top mu if we have at least mu finite values
            n_use = min(mu, gen_pop)
            # Check how many are finite
            finite_mask = np.isfinite(sorted_fvals[:n_use])
            if not np.all(finite_mask):
                n_use = np.sum(finite_mask)
                if n_use == 0:
                    break

            y_w = np.sum(weights[:n_use][:, None] * sorted_offspring[:n_use], axis=0)
            m = np.sum(weights[:n_use])
            y_w = y_w / m

            # Update mean
            x_new = x + y_w

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                C_sqrt_inv @ y_w / sigma
            )
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1))) < (
                1.4 + 2 / (n + 1)
            ) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(
                c_c * (2 - c_c) * mueff
            ) * y_w / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            diff = (x_new - x) / sigma
            C = (
                (1 - c_1 - c_mu + delta_hsig * c_c * (2 - c_c)) * C
                + c_1 * (np.outer(pc, pc) - (1 - c_c) * C)
                + c_mu * np.sum(
                    weights[:n_use][:, None, None]
                    * np.outer(sorted_offspring[:n_use] - x, sorted_offspring[:n_use] - x)
                    / sigma ** 2,
                    axis=0,
                )
            )

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definiteness by adding small identity if needed
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

            C_sqrt_inv = np.linalg.inv(L)

            # Update sigma
            sigma *= np.exp(
                c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)
            )
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 100.0)

            x = x_new

            if remaining <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
