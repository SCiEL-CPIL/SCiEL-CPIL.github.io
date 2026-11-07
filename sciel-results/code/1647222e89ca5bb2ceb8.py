"""Optimized search for BBOB f14 (Different powers) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Global-Local hybrid optimizer:
    Phase 1: CMA-ES style population-based search for global exploration.
    Phase 2: (1+1)-ES with adaptive step size for local polishing.
    Includes a deadline guard to stay within 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320.0  # 10s safety margin before 330s

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Population-based global search (CMA-ES-like) ---
        # Use a moderate population size for 5D problem
        pop_size = 12
        # Initial center and sigma
        center = np.zeros(dim)
        sigma = 2.5

        # Initialize population
        population = center + sigma * rng.standard_normal((pop_size, dim))
        population = np.clip(population, -5, 5)
        fitnesses = np.array([func(x) for x in population])
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Track best-so-far for AOCC
        calls_made = pop_size
        b_list = [best_f]

        # CMA-ES parameters
        c1 = (4 + 3 / dim) / ((dim + 4) ** 2 + 4)
        cmu = 2 * (min(2 - c1, 0.5 / (dim + 1)) - c1) / ((dim + 2) * (dim + 2 + 2))
        cs = (dim + 1.7) / (dim + 9 + mu_eff) if (dim + 9 + mu_eff) != 0 else 0.5
        # Simpler: use fixed damping
        cs = 1.0 / (dim + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + 1)
        cmu = min(1 - c1, 2 * (min(2 - c1, 0.5 / (dim + 1)) - c1) / ((dim + 2) * (dim + 2 + 2)))
        damps = 1 + 2 * max(0, sqrt((dim - 1) * (dim - 3)) / (dim + 2) - 1) + cs

        def sqrt(x):
            return np.sqrt(x)

        # Recompute with correct sqrt
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) * (dim - 3)) / (dim + 2) - 1) + cs

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Phase 1: iterate until budget or time limit
        phase1_budget = int(budget * 0.6)
        for gen in range(phase1_budget // pop_size + 1):
            if time.time() > self.deadline:
                break
            if calls_made >= budget:
                break

            # Sort population by fitness
            order = np.argsort(fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]

            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0].copy()
            b_list.append(best_f)

            # Compute mean of parents
            m_new = np.sum(weights[:, None] * population[:mu], axis=0)

            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.inv(np.sqrt(C)) @ (m_new - center) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) / np.sqrt(mu_eff) < 1.4 + 2 / (dim + 1)
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - center) / sigma

            # Update covariance matrix
            delta = 1 + 2 * max(0, sqrt((dim - 1) * (dim - 3)) / (dim + 2) - 1) + cs
            C = (1 + c1 * (2 - c1) - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / delta * np.sum(
                weights[:, None, None] * np.outer(population[:mu] - m_new, population[:mu] - m_new) / sigma ** 2, axis=0
            )

            # Symmetrize
            C = (C + C.T) / 2

            # Update center and sigma
            center = m_new
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 50)

            # Generate new population
            if calls_made + pop_size > budget:
                # Only generate remaining
                remaining = budget - calls_made
                new_pop = center + sigma * rng.standard_normal((remaining, dim))
                new_pop = np.clip(new_pop, -5, 5)
                new_fit = np.array([func(x) for x in new_pop])
                calls_made += remaining
                for f in new_fit:
                    if f < best_f:
                        best_f = f
                    b_list.append(best_f)
                break

            new_pop = center + sigma * rng.standard_normal((pop_size, dim))
            new_pop = np.clip(new_pop, -5, 5)
            new_fit = np.array([func(x) for x in new_pop])
            calls_made += pop_size
            for f in new_fit:
                if f < best_f:
                    best_f = f
                b_list.append(best_f)

            population = new_pop
            fitnesses = new_fit

        # --- Phase 2: (1+1)-ES local polishing ---
        remaining = budget - calls_made
        if remaining > 0 and time.time() < self.deadline:
            x = best_x.copy()
            fx = best_f
            sigma = max(sigma * 0.5, 0.1)

            for _ in range(remaining):
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    best_f = fx
                    best_x = x
                else:
                    sigma *= 0.9
                b_list.append(best_f)

        # Ensure b_list has at least budget entries
        while len(b_list) < budget:
            b_list.append(b_list[-1] if b_list else 0.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
