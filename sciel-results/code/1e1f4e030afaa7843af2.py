"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+lambda)-ES with adaptive step size for f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Phase 1: Global exploration - evaluate many random points
        n_init = min(50, budget // 10)
        init_budget = n_init

        candidates = []
        for i in range(n_init):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            candidates.append((fx, x))

        # Sort by fitness and keep top K
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        elites = [x for _, x in candidates[:top_k]]

        # Phase 2: Local exploitation with (1+lambda)-ES from each elite
        remaining_budget = budget - init_budget
        lambda_ = 8  # number of offspring per generation
        budget_per_elite = remaining_budget // top_k

        best_x = elites[0]
        best_f = candidates[0][0]

        for elite_idx in range(top_k):
            if time.time() > deadline:
                break

            x = elites[elite_idx].copy()
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            successes = 0
            generations = 0

            while generations < budget_per_elite and time.time() < deadline:
                # Generate lambda_ offspring
                offspring = []
                for j in range(lambda_):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    offspring.append((fy, y))

                generations += 1

                # Select best offspring
                offspring.sort(key=lambda t: t[0])
                best_fy, best_y = offspring[0]

                # Count successes
                successes = sum(1 for fy, _ in offspring if fy < fx)

                # Update if improvement
                if best_fy < fx:
                    x = best_y
                    fx = best_fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Adaptive step size: target 20% success rate
                target = 0.2
                if successes > target * lambda_:
                    sigma *= 1.1
                elif successes < target * lambda_:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
