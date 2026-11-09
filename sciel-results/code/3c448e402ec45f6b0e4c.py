"""Rewritten optimizer for BBOB-style Schwefel f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    Uses a (mu+lambda)-ES with adaptive step size, multiple restarts,
    and a deadline guard to stay within the wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track best point and value globally
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Use a (mu+lambda)-ES with mu=4, lambda=20
        mu = 4
        lam = 20
        sigma = 2.0

        # Number of restarts allowed
        max_restarts = 5
        stagnation_limit = 50  # restart if no improvement in this many iterations

        def run_es(start_x, start_sigma, max_iters):
            nonlocal total_calls, best_x, best_f

            # Initialize population
            pop = []
            for i in range(mu):
                x = np.clip(start_x + start_sigma * rng.standard_normal(dim), -5, 5)
                if total_calls >= budget:
                    break
                fx = func(x)
                total_calls += 1
                pop.append((x, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Sort by fitness
            pop.sort(key=lambda t: t[1])

            iters = 0
            no_improve = 0
            prev_best = best_f

            for _ in range(max_iters):
                if total_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Generate offspring
                offspring = []
                for i in range(lam):
                    parent = pop[i % mu][0]
                    x = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                    if total_calls >= budget:
                        break
                    fx = func(x)
                    total_calls += 1
                    offspring.append((x, fx))
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                if not offspring:
                    break

                # Merge population and offspring, select best mu
                all_ind = pop + offspring
                all_ind.sort(key=lambda t: t[1])
                pop = all_ind[:mu]

                # Adapt sigma based on success rate
                # Count how many offspring were better than their parent
                successes = 0
                for i in range(min(lam, len(offspring))):
                    if offspring[i][1] < pop[min(i, mu - 1)][1]:
                        successes += 1
                rate = successes / max(1, min(lam, len(offspring)))
                # 1/5 rule
                if rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

                iters += 1
                if best_f < prev_best - 1e-6:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1

                if no_improve >= stagnation_limit:
                    break

            return iters, no_improve

        # Phase 1: Initial exploration with diverse starting points
        # Start with a few diverse points
        n_initial = 5
        for i in range(n_initial):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5:
                break

            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Run ES from the best point found
        if best_x is not None:
            iters, no_improve = run_es(best_x, sigma, budget - total_calls)

            # If stagnation occurred and we have budget left, try restarts
            restarts = 0
            while (no_improve >= stagnation_limit and
                   total_calls < budget and
                   time.time() < deadline - 10 and
                   restarts < max_restarts):
                # Try a new random start
                x_new = rng.uniform(-5, 5, dim)
                if total_calls >= budget:
                    break
                fx_new = func(x_new)
                total_calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()

                # Run ES from this new point
                iters, no_improve = run_es(best_x, sigma, budget - total_calls)
                restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
