"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Multi-start: use a 3^dim grid for initial points, plus random
        # 3^5 = 243 points, which is a good fraction of 500 budget
        grid_vals = np.array([-4.0, 0.0, 4.0])
        starts = []
        for i in range(3 ** dim):
            pt = np.array([grid_vals[(i // (3 ** d)) % 3] for d in range(dim)])
            starts.append(pt)

        # Add a few random starts
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        for s in starts:
            if calls >= budget or time_left() < 0.5:
                break
            f = eval_point(s)
            if f < best_f:
                best_f = f
                best_x = s.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)

        # CMA-ES parameters
        def cma_es(x0, f0, remaining_budget, max_time):
            nonlocal calls
            dim = self.dim
            mu = max(1, int(np.floor(self.dim / 2)))
            lam = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            sigma = 0.5
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            C_inv = np.eye(dim)
            # Precompute sqrt of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Track population for diversity
            pop = [(x0.copy(), f0)]

            while calls < budget and time_left() > max(1.0, 0.01 * max_time):
                if calls + lam > budget:
                    lam = max(2, budget - calls)
                    mu = max(1, lam // 2)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mu_eff = 1.0 / np.sum(weights ** 2)

                # Generate offspring
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x0 + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offs_f = []
                for y in offspring:
                    if calls >= budget or time_left() < 0.5:
                        break
                    fy = eval_point(y)
                    offs_f.append(fy)

                if len(offs_f) < 2:
                    break

                # Sort by fitness
                order = np.argsort(offs_f)
                offspring_sorted = [offspring[i] for i in order[:mu]]
                offs_f_sorted = [offs_f[i] for i in order[:mu]]

                # Update best
                if offs_f_sorted[0] < best_f:
                    best_f = offs_f_sorted[0]
                    best_x = offspring_sorted[0].copy()

                # Update population
                pop = list(zip(offspring_sorted, offs_f_sorted))

                # Mean of selected
                m_new = np.sum(np.array(offspring_sorted) * weights[:, None], axis=0)

                # Update evolution paths
                diff = m_new - x0
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ diff) / sigma
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - len(pop)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1):
                    pc = (1 - c_c) * pc + c_c * (diff / sigma - pc)
                    delta = 1 - (1 - c_c) ** 2
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta) + c_mu * np.sum(weights[:, None, None] * np.outer(np.array(offspring_sorted) - m_new, np.array(offspring_sorted) - m_new), axis=0)

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                C_inv = np.linalg.inv(C)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim + 1) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x0 = m_new

                # Check for stagnation - if no improvement in last 50 calls, restart
                # We'll handle restarts at the outer level

            return x0, best_f, best_x

        # Do CMA-ES from the best start
        remaining = budget - calls
        max_time = time_left()
        if remaining > 20 and max_time > 5:
            x0 = best_x.copy()
            f0 = best_f
            _, _, best_x = cma_es(x0, f0, remaining, max_time)

        # If we still have budget, do a second CMA-ES from a different start
        if calls < budget - 10 and time_left() > 10:
            # Pick a different start
            alt_idx = 0
            for i, s in enumerate(starts):
                if not np.allclose(s, best_x, atol=0.1):
                    alt_idx = i
                    break
            x0 = starts[alt_idx].copy()
            f0 = eval_point(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            remaining = budget - calls
            max_time = time_left()
            if remaining > 10 and max_time > 5:
                _, _, best_x = cma_es(x0, f0, remaining, max_time)

        return best_x, best_f
# EVOLVE-BLOCK-END
