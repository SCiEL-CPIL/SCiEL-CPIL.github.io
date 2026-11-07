"""Multi-start (1+λ)-ES with population diversity for Büche-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+λ)-ES with population, adaptive sigma, and multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Population size for (1+λ)-ES
        lam = 6  # offspring per generation
        # Initial population of elites
        pop_size = 3

        # Initialize population with diverse points
        best_x = None
        best_f = np.inf
        population = []  # list of (f, x)

        # Evaluate initial diverse points
        init_points = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            init_points.append(rng.uniform(-4, 4, size=dim))

        for x in init_points:
            f = func(np.clip(x, -5, 5))
            population.append((f, np.clip(x, -5, 5)))
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()

        calls_used = len(init_points)

        sigma = 1.5  # Start with larger step for exploration
        sigma_min = 1e-10
        sigma_max = 8.0

        # Adaptive sigma parameters
        success_window = 0
        success_count = 0
        window_size = 8  # smaller window for faster adaptation

        # Stagnation detection
        no_improve_calls = 0
        max_no_improve = 30
        restarts = 0
        max_restarts = 8

        while calls_used < budget and time.time() < deadline:
            # Generate λ offspring from the best point in population
            # Use the best point as parent, but also occasionally from other elites
            if rng.random() < 0.7:
                parent = best_x
            else:
                # Pick a random elite from population
                _, parent = population[rng.integers(0, len(population))]

            # Generate offspring
            offspring = []
            for _ in range(lam):
                step = sigma * rng.standard_normal(dim)
                trial_x = np.clip(parent + step, -5, 5)
                trial_f = func(trial_x)
                calls_used += 1
                offspring.append((trial_f, trial_x))

                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x.copy()

            # Keep top pop_size from population + offspring
            all_candidates = population + offspring
            all_candidates.sort(key=lambda t: t[0])
            population = all_candidates[:pop_size]

            # Track success for sigma adaptation
            success_window += 1
            if any(f < best_f for f, _ in offspring):
                success_count += 1

            # Adapt sigma
            if success_window >= window_size:
                rate = success_count / success_window
                if rate > 0.5:
                    sigma *= 1.3
                elif rate < 0.1:
                    sigma /= 1.3
                else:
                    sigma *= 1.05
                sigma = max(sigma, sigma_min)
                sigma = min(sigma, sigma_max)
                success_count = 0
                success_window = 0

            # Stagnation check
            if best_f < population[0][0] + 1e-12:
                no_improve_calls += 1
            else:
                no_improve_calls = 0

            # Actually track no-improvement properly
            # Reset if we improved
            if len(offspring) > 0 and min(f for f, _ in offspring) < best_f + 1e-12:
                no_improve_calls = 0

            if no_improve_calls >= max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break

                # Multi-start restart: re-seed population with new random points
                sigma = max(1.0, sigma * 0.5)
                no_improve_calls = 0
                new_pop = [(best_f, best_x.copy())]
                for _ in range(pop_size - 1):
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    cand_x = rng.uniform(-4, 4, size=dim)
                    cand_f = func(cand_x)
                    calls_used += 1
                    new_pop.append((cand_f, cand_x))
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand_x.copy()
                population = new_pop

        return best_x, best_f
# EVOLVE-BLOCK-END
