"""Rewritten optimizer for BBOB f12 (Bent cigar) with black-box budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(d)
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start with different strategies
        # Phase 1: Broad exploration with random restarts
        # Phase 2: Adaptive step-size local search around best
        # Phase 3: CMA-ES-like covariance adaptation

        # Initial broad sampling
        n_init = min(20, budget)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, d)
            eval_point(x)

        # Main loop: adaptive local search with population
        # Use a population-based approach with adaptive step sizes
        pop_size = min(8, max(4, budget // 50))
        population = []

        # Initialize population around best and random points
        for i in range(pop_size):
            if i == 0:
                x = best_x + rng.normal(0, 1.0, d)
            else:
                x = rng.uniform(-5, 5, d)
            x = np.clip(x, -5, 5)
            fx = eval_point(x)
            population.append((x.copy(), fx))

        # Sort population
        population.sort(key=lambda p: p[1])

        # Adaptive step sizes per dimension
        sigma = 1.0
        step_sizes = np.ones(d)

        stagnation_counter = 0
        max_stagnation = 30

        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(pop_size):
                parent_x, parent_f = population[i]

                # Adaptive mutation: larger for worse parents
                scale = 1.0 + 0.5 * (i / pop_size)
                mutation = sigma * scale * rng.standard_normal(d) * step_sizes
                child_x = np.clip(parent_x + mutation, -5, 5)
                child_f = eval_point(child_x)
                offspring.append((child_x.copy(), child_f))

            # Combine and select
            combined = population + offspring
            combined.sort(key=lambda p: p[1])
            population = combined[:pop_size]

            # Update best
            if population[0][1] < best_f:
                best_f = population[0][1]
                best_x = population[0][0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adapt step size based on success rate
            # Count how many offspring improved their parents
            improvements = 0
            for i in range(pop_size):
                parent_f = population[i][1]
                # Check if any offspring beat the parent
                for ox, of in offspring:
                    if of < parent_f:
                        improvements += 1
                        break

            success_rate = improvements / pop_size
            if success_rate > 0.3:
                sigma *= 1.1
            elif success_rate < 0.1:
                sigma *= 0.8
            sigma = np.clip(sigma, 0.01, 5.0)

            # Adapt step sizes per dimension based on coordinate changes
            if call_count % 10 == 0:
                # Estimate coordinate importance from recent improvements
                pass  # Keep simple for now

            # Stagnation handling: restart with new random point
            if stagnation_counter > max_stagnation:
                # Inject a new random point into population
                x_new = rng.uniform(-5, 5, d)
                fx_new = eval_point(x_new)
                population.append((x_new.copy(), fx_new))
                population.sort(key=lambda p: p[1])
                population = population[:pop_size]
                stagnation_counter = 0
                max_stagnation = max(15, max_stagnation // 2)

            # Occasionally do a finer local search around best
            if call_count % 50 == 0 and call_count > 100:
                for _ in range(min(5, budget - call_count)):
                    x_local = best_x + rng.normal(0, sigma * 0.3, d)
                    x_local = np.clip(x_local, -5, 5)
                    eval_point(x_local)

        # Final: use remaining budget for fine-tuning around best
        while call_count < budget:
            if time.time() > deadline:
                break
            x_fine = best_x + rng.normal(0, max(0.01, sigma * 0.1), d)
            x_fine = np.clip(x_fine, -5, 5)
            eval_point(x_fine)

        return best_x, best_f
# EVOLVE-BLOCK-END
