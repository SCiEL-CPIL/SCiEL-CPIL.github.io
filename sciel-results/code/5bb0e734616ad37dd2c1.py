"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a random point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        # Adaptive step size
        sigma = 1.5
        lambda_ = 10  # number of offspring per generation

        # Track best
        best_x = x.copy()
        best_f = fx

        # Success counter for 1/5 rule
        successes = 0
        attempts = 0

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            best_off_f, best_off_x = offspring[0]

            attempts += 1

            if best_off_f < fx:
                # Accept the best offspring
                x = best_off_x
                fx = best_off_f
                successes += 1

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Adapt sigma (increase on success)
                sigma *= 1.2
            else:
                # Reject, decrease sigma
                sigma *= 0.8

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            # Occasionally try a larger jump to escape local plateaus
            if attempts % 50 == 0:
                y = np.clip(x + 2.0 * sigma * rng.standard_normal(dim), -5, 5)
                if calls < budget and time.time() < deadline:
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
