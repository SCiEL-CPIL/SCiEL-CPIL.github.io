"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (1+λ)-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Weierstrass f16 is known to be multimodal with a relatively flat
        # landscape. A population-based approach with adaptive step sizes
        # should work well.

        # Initial population of diverse points
        pop_size = 10
        # Start with a broad exploration
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls = 1

        # Adaptive sigma
        sigma = 1.5

        # Track success rate for sigma adaptation
        successes = 0
        total_trials = 0

        while calls < budget:
            # Deadline check
            if time.time() > deadline:
                break

            # Generate offspring
            n_offspring = 20
            for _ in range(n_offspring):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Adaptive mutation
                y = x_best + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < f_best:
                    x_best = y
                    f_best = fy
                    successes += 1

            total_trials += n_offspring

            # Adapt sigma based on success rate (target ~1/5)
            if total_trials > 0:
                success_rate = successes / total_trials
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 1.05

            # Reset counters
            successes = 0
            total_trials = 0

            # Periodic restart: if stuck, try a new random point
            if calls % 50 == 0:
                x_restart = rng.uniform(-5, 5, dim)
                f_restart = func(x_restart)
                calls += 1
                if f_restart < f_best:
                    x_best = x_restart
                    f_best = f_restart

        return x_best, f_best
# EVOLVE-BLOCK-END
