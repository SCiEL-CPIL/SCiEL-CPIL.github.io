"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive sigma for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: evaluator kills after 330s. Allow 280s for safety.
        start_time = time.time()
        max_time = 280.0

        rng = np.random.default_rng(42)

        # Population size
        mu = 4
        lam = 12

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        sigma_min = 1e-4
        sigma_max = 10.0

        # Track success rate for sigma adaptation
        success_count = 0
        window = 20  # adaptation window

        calls_used = 1

        while calls_used < budget:
            # Check time
            if time.time() - start_time > max_time:
                break

            # Generate offspring
            offspring = []
            for _ in range(lam):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((y, fy))

                # Check time inside loop too
                if calls_used >= budget or time.time() - start_time > max_time:
                    break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])

            # Take best mu offspring
            if len(offspring) < mu:
                # Not enough offspring generated (time/budget ran out)
                break

            selected = offspring[:mu]

            # Adapt sigma based on success rate
            # Count how many selected offspring improved over parent
            improved = sum(1 for (_, fy) in selected if fy < fx)
            success_rate = improved / mu

            # 1/5 rule: if success rate > 0.2, increase sigma; else decrease
            if success_rate > 0.2:
                sigma *= 1.3
            elif success_rate < 0.2:
                sigma *= 0.8
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Update parent as mean of selected offspring
            x_new = np.mean([pt for (pt, _) in selected], axis=0)
            fx_new = min(fy for (_, fy) in selected)

            # Update best
            if fx_new < best_fx:
                best_x = x_new.copy()
                best_fx = fx_new

            x, fx = x_new, fx_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
