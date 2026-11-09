"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase population-based optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse random initialization - sample many points to find good basin
        # Weierstrass has many local minima, so global exploration is critical
        n_coarse = min(100, budget // 5)
        for i in range(n_coarse):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            update_best(x, f)

        # Phase 2: Population-based local search (CMA-ES-like with restarts)
        # Use a population of candidates with adaptive step sizes
        pop_size = 8
        remaining = budget - n_coarse
        if remaining <= 0:
            return best_x, best_f

        # Initialize population around best found + random perturbations
        population = []
        for i in range(pop_size):
            if i == 0:
                x = best_x.copy()
            else:
                # Mix of local and global perturbations
                if rng.random() < 0.5:
                    x = best_x + rng.normal(0, 1.5, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)
            population.append((x, f))

        remaining -= pop_size
        if remaining <= 0:
            return best_x, best_f

        # Adaptive step size for population
        sigma = 1.0
        best_idx = 0
        population[best_idx] = (best_x, best_f)

        # Main loop: population-based search with adaptive steps
        while remaining > 0 and time.time() < deadline:
            # Generate offspring for each parent
            offspring = []
            for i in range(min(pop_size, remaining)):
                parent_x, parent_f = population[i]
                # Adaptive perturbation
                scale = sigma * (1.0 + 0.5 * rng.random())
                child = parent_x + scale * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                f = func(child)
                update_best(child, f)
                offspring.append((child, f))
                remaining -= 1

            if not offspring:
                break

            # Selection: keep best half of population + best offspring
            all_candidates = population + offspring
            all_candidates.sort(key=lambda t: t[1])
            population = all_candidates[:pop_size]

            # Adapt sigma based on improvement
            best_offspring_f = min(f for _, f in offspring)
            prev_best_f = min(f for _, f in population[:-len(offspring)]) if len(offspring) < len(population) else np.inf
            if best_offspring_f < prev_best_f:
                sigma *= 1.3
            else:
                sigma *= 0.9

            # Periodic restart: if stuck, inject diversity
            if remaining % 20 == 0 and remaining > 0:
                # Check if we've improved recently
                if sigma < 0.1:
                    # Inject a few global random points
                    for _ in range(min(3, remaining)):
                        x = rng.uniform(-5, 5, dim)
                        f = func(x)
                        update_best(x, f)
                        population.append((x, f))
                        remaining -= 1
                    population.sort(key=lambda t: t[1])
                    population = population[:pop_size]
                    sigma = 1.0

        # Phase 3: Fine-tuning around best with very small steps
        if remaining > 0 and best_x is not None and time.time() < deadline:
            fine_sigma = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                child = best_x + fine_sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                f = func(child)
                if f < best_f:
                    best_f = f
                    best_x = child
                    fine_sigma *= 1.2
                else:
                    fine_sigma *= 0.8
                remaining -= 1

        return best_x, best_f
# EVOLVE-BLOCK-END
