import numpy as np
import time


class Optimizer:
    """Population-based ES with adaptive step size for multi-modal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 290.0  # Safety margin before 330s kill

        # Parameters
        lam = 20  # offspring per generation
        mu = 5    # number of parents
        sigma = 1.0
        min_sigma = 1e-8
        max_sigma = 10.0

        # Start at origin (often near optimum for shifted/rotated functions)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Generation loop
        no_improve = 0
        gen = 0

        while calls < budget:
            # Deadline check
            if time.time() - start_time > max_time:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if time.time() - start_time > max_time:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update with best offspring
            best_off_f, best_off_x = offspring[0]
            if best_off_f < fx:
                fx = best_off_f
                x = best_off_x
                no_improve = 0
            else:
                no_improve += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma: increase if many improvements, decrease otherwise
            improved = sum(1 for f, _ in offspring if f < fx + 1e-10)
            success_rate = improved / len(offspring)
            if success_rate > 0.5:
                sigma = min(sigma * 1.3, max_sigma)
            elif success_rate < 0.2:
                sigma = max(sigma * 0.7, min_sigma)

            # Restart strategy: if stuck for too long, increase sigma or random restart
            if no_improve >= 5:
                if no_improve >= 10:
                    # Random restart from a new point
                    x = np.random.uniform(-4, 4, dim)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 2.0
                    no_improve = 0
                else:
                    # Increase sigma to escape local minimum
                    sigma = min(sigma * 2.0, max_sigma)

            gen += 1

        return best_x, best_f
