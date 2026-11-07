"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Initial population: diverse starting points
        rng = np.random.default_rng(42)

        # Start with a population of size 4
        pop_size = 4
        sigma = 1.5

        # Initialize population
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)

        # Create initial population around best
        population = [x_best.copy()]
        for i in range(pop_size - 1):
            x_new = np.clip(x_best + sigma * rng.standard_normal(dim), -5, 5)
            population.append(x_new)

        # Evaluate rest of population
        fvals = [f_best]
        for i in range(1, pop_size):
            fi = func(population[i])
            fvals.append(fi)
            if fi < f_best:
                f_best = fi
                x_best = population[i].copy()

        calls_used = budget - 1  # we used budget-1 so far (1 initial + pop_size-1)
        # Actually: 1 (initial) + (pop_size-1) = pop_size calls
        calls_used = pop_size

        # Main loop
        while calls_used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            # Use a mix of local and global exploration
            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Adaptive sigma: shrink when improving, expand when stuck
                # Generate candidate
                if i == 0:
                    # Best point: small perturbation
                    step = sigma * 0.5
                elif i == 1:
                    # Medium exploration
                    step = sigma
                else:
                    # Larger exploration
                    step = sigma * 1.5

                x_new = np.clip(x_best + step * rng.standard_normal(dim), -5, 5)
                f_new = func(x_new)
                calls_used += 1

                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
                    # Successful step: increase sigma slightly
                    sigma = min(sigma * 1.1, 4.0)
                else:
                    # Unsuccessful: decrease sigma
                    sigma = max(sigma * 0.95, 0.1)

            # Occasionally do a random restart if stuck
            if calls_used % (pop_size * 50) == 0:
                # Try a few random points
                for _ in range(min(3, budget - calls_used)):
                    if time.time() > deadline:
                        break
                    x_rand = rng.uniform(-5, 5, dim)
                    f_rand = func(x_rand)
                    calls_used += 1
                    if f_rand < f_best:
                        f_best = f_rand
                        x_best = x_rand.copy()
                        sigma = 1.5

        return x_best, f_best
# EVOLVE-BLOCK-END
