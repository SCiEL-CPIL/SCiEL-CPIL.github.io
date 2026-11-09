"""Rewritten optimizer: population-based adaptive search with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive search with step-size annealing and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize population of 10 candidates
        pop_size = 10
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_made = pop_size
        sigma = 2.0  # initial step size
        no_improve_count = 0

        # Main adaptive population search
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Evaluate improvement rate
            current_best = fvals.min()
            if current_best < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Anneal sigma based on stagnation
            if no_improve_count > 20:
                sigma *= 0.7
                no_improve_count = 0
            elif no_improve_count < 5:
                sigma *= 1.1

            sigma = max(0.01, min(sigma, 5.0))

            # Generate offspring from each parent
            improved = False
            new_fvals = fvals.copy()
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Mutate from parent i
                y = np.clip(pop[i] + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                # Update population: keep better of parent and offspring
                if fy < fvals[i]:
                    pop[i] = y
                    fvals[i] = fy
                    improved = True

                    # Update global best
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy

            # Occasionally replace worst with new random point to maintain diversity
            if calls_made < budget and rng.random() < 0.1:
                worst_idx = np.argmax(fvals)
                new_point = rng.uniform(-5, 5, dim)
                fy = func(new_point)
                calls_made += 1
                pop[worst_idx] = new_point
                fvals[worst_idx] = fy
                if fy < best_f:
                    best_x = new_point.copy()
                    best_f = fy

        # Final local refinement on the best candidate
        if calls_made < budget and time.time() < deadline:
            local_sigma = 0.5
            for _ in range(20):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                local_sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
