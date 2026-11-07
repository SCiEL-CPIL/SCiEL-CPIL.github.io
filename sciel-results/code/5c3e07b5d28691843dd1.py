"""Rewritten optimizer: multi-restart CMA-ES with time guard for BBOB f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for 5-D Schwefel-like minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_used = 0

        # Multi-restart strategy
        n_restarts = 4
        per_restart = max(100, budget // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline - 10.0 or calls_used >= budget:
                break

            rng = np.random.default_rng(12345 + restart * 7919)

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            elif np.isfinite(best_f) and rng.random() < 0.4:
                x = np.clip(best_x + rng.uniform(-1.5, 1.5, dim), -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES setup
            sigma = 2.0
            pop_size = dim + 3  # 8 for dim=5
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_sigma = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)

            local_best_x = x.copy()
            local_best_f = fx
            local_calls = 1
            max_iter = min(per_restart - local_calls, budget - calls_used)

            for iteration in range(max_iter):
                if time.time() > deadline - 5.0 or calls_used >= budget:
                    break

                # Generate population
                offspring = []
                for _ in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = np.array([func(y) for y in offspring])
                calls_used += pop_size
                for i, (y, fy) in enumerate(zip(offspring, offspring_f)):
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                # Sort
                order = np.argsort(offspring_f)
                offspring_sorted = np.array(offspring)[order]
                offspring_f_sorted = offspring_f[order]

                # Update mean
                x_new = np.sum(weights[:, None] * offspring_sorted[:mu], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update path states
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * (x_new - x) / sigma))
                ps_norm = np.linalg.norm(ps)
                hc = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_sigma else 0.0
                pc = (1 - c_c) * pc + hc * c_c * (x_new - x) / sigma

                # Update C
                delta_hc = (1 - hc) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_mu) * C  # base
                # Rank-one update
                C += c_1 * (pc @ pc.T - C)  # approximate
                # Rank-mu update
                for i in range(mu):
                    y_i = (offspring_sorted[i] - x) / sigma
                    C += c_mu * (y_i @ y_i.T)
                C -= (c_1 * np.eye(dim) + c_mu * np.eye(dim))  # correction
                # Ensure symmetry
                C = 0.5 * (C + C.T)

                # Update B and D
                D2, B = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-12)
                D = np.sqrt(D2)

                # Update sigma
                sigma *= np.exp(c_sigma / chi_sigma * (ps_norm / chi_sigma - 1))
                sigma = np.clip(sigma, 0.005, 15.0)

                x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
