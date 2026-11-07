"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start for multimodal f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with Latin Hypercube Sampling
        n_global = 60
        samples = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points

        f_samples = np.array([func(x) for x in samples])
        calls = n_global

        if time.time() - start_time > max_time:
            idx = np.argmin(f_samples)
            return samples[idx], f_samples[idx]

        sorted_idx = np.argsort(f_samples)
        best_x = samples[sorted_idx[0]].copy()
        best_f = f_samples[sorted_idx[0]]

        # Phase 2: Multi-start CMA-ES from top candidates
        n_starts = 5
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Reserve a small safety budget
        safety = max(10, int(remaining * 0.03))
        alloc = remaining - safety
        budget_per_start = max(20, alloc // n_starts)

        for k in range(n_starts):
            if calls >= budget or time.time() - start_time > max_time:
                break

            x0 = samples[sorted_idx[k]].copy()
            f0 = f_samples[sorted_idx[k]]

            # CMA-ES parameters
            pop_size = 10
            x = x0.copy()
            sigma = 0.8
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            mu = pop_size // 2
            weights = np.ones(mu)
            weights -= np.arange(mu) * 0.1
            weights = np.maximum(weights, 0.05)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

            local_budget = min(budget_per_start, budget - calls)
            no_improve_count = 0
            local_best_f = f0

            for gen in range(local_budget // pop_size):
                if calls >= budget or time.time() - start_time > max_time:
                    break

                # Generate population
                offspring = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    # Use Cholesky-like update (simple: use C_sqrt approximation)
                    y = C_sqrt @ z
                    offspring[i] = x + sigma * y

                # Evaluate
                f_offspring = np.array([func(offspring[i]) for i in range(pop_size)])
                calls += pop_size

                if calls > budget:
                    break

                # Sort offspring
                order = np.argsort(f_offspring)
                selected = offspring[order[:mu]]
                f_selected = f_offspring[order[:mu]]

                # Update best
                if f_selected[0] < local_best_f:
                    local_best_f = f_selected[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Restart if stagnated
                if no_improve_count > 15:
                    break

                # Weighted mean
                m_new = np.sum(weights[:, None] * selected, axis=0)

                # Update evolution paths
                diff = m_new - x
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (diff / sigma))
                h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0.0
                pc = (1 - c_c) * pc + h_sigma * c_c / sigma * diff

                # Update covariance
                delta = (1 + h_sigma * c_c * (2 - c_c)) * np.outer(pc, pc) - np.eye(dim)
                C = (1 + c_1 + c_mu - c_1) * C + c_1 * delta + c_mu * np.sum(weights[:, None, None] * np.einsum('i,ij,ik->jk', weights, selected - m_new, selected - m_new), axis=0)

                # Regularize C
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-6, 5.0)

                # Update mean
                x = m_new

                # Update C_sqrt (recompute each gen for simplicity)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

                # Track global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = selected[0].copy()

        # Phase 3: Final refinement with fine-grained (1+1)-ES
        if calls < budget and time.time() - start_time < max_time:
            x = best_x.copy()
            fx = best_f
            final_budget = budget - calls
            sigma = 0.05

            for _ in range(final_budget):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.92
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
