"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + (lambda+mu)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration to find a promising basin
        n_starts = 30
        best_x = None
        best_fx = np.inf
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Population-based (lambda+mu)-ES with success-rule adaptation
        x = best_x
        fx = best_fx
        sigma = 1.0
        lam = 8  # offspring population size
        mu = 1   # keep best 1
        remaining = budget - n_starts

        for gen in range(remaining // lam):
            if time.time() > deadline:
                break
            # Generate lambda offspring
            offspring = []
            for _ in range(lam):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                if time.time() > deadline:
                    break
            if len(offspring) == 0:
                break
            offspring.sort(key=lambda t: t[0])
            best_fy, best_y = offspring[0]
            if best_fy < fx:
                x, fx = best_y, best_fy
                sigma *= 1.3  # success: increase step
            else:
                sigma *= 0.7  # failure: decrease step
            sigma = max(sigma, 0.01)

        return x, fx
# EVOLVE-BLOCK-END
