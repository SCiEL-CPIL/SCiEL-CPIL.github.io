"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (mu+lambda)-ES with adaptive step size and restarts for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Broad random search to find a decent starting point
        n_init = min(15, budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: Population-based ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Population size: small for dim=5 to conserve budget
        lambda_ = 6
        mu = 3

        calls_used = n_init
        no_improve_count = 0
        max_no_improve = 30  # restart after this many evaluations without improvement

        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for j in range(lambda_):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_used += 1

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Count successes against old best before updating
            n_success = sum(1 for fy, _ in offspring if fy < fx)

            # Update best if improvement found
            if offspring[0][0] < fx:
                fx = offspring[0][0]
                x = offspring[0][1].copy()
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += lambda_

            # Step size adaptation: increase if success rate > 1/5, decrease otherwise
            success_rate = n_success / len(offspring)
            if success_rate > 0.2:
                sigma *= 1.15
            else:
                sigma *= 0.9

            sigma = np.clip(sigma, 0.005, 5.0)

            # Restart if stuck
            if no_improve_count >= max_no_improve and calls_used < budget * 0.85:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_used += 1
                sigma = 1.0
                no_improve_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final polish with smaller sigma if budget remains
        if calls_used < budget and time.time() < deadline:
            sigma = max(0.005, sigma * 0.2)
            while calls_used < budget and time.time() < deadline:
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma = max(0.001, sigma * 1.1)
                else:
                    sigma *= 0.85

        return (best_x, best_f)
# EVOLVE-BLOCK-END
