"""Optimized BBOB f6 solver using adaptive (1+lambda) ES with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+lambda) Evolution Strategy with multi-start for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Define a helper to run a local (1+lambda) ES search
        def run_es(x_start, sigma_init, budget_alloc, time_limit):
            nonlocal total_calls, best_x, best_f
            
            x = x_start.copy()
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if total_calls >= budget or time.time() > time_limit:
                return

            sigma = sigma_init
            sigma_min = 1e-5
            sigma_max = 10.0
            
            # Adaptive lambda: start large, shrink as sigma shrinks
            # This allows more iterations (generations) in the later stages
            # where we need to make small, precise moves.
            
            calls_in_es = 1
            while calls_in_es < budget_alloc and total_calls < budget and time.time() < time_limit:
                # Determine population size based on current sigma
                # If sigma is large, we can afford larger lambda to explore.
                # If sigma is small, we want to use the budget for more generations (smaller lambda).
                if sigma > 1.0:
                    lam = 12
                elif sigma > 0.1:
                    lam = 6
                else:
                    lam = 3

                # Check if we can afford another generation
                if total_calls + lam > budget:
                    lam = budget - total_calls
                    if lam <= 0:
                        break

                # Generate offspring
                offspring = []
                for _ in range(lam):
                    if time.time() > time_limit:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    calls_in_es += 1
                    offspring.append((fy, y))

                if not offspring:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                best_fy, best_y = offspring[0]

                if best_fy < fx:
                    # Success: move to best, increase sigma
                    x = best_y
                    fx = best_fy
                    sigma = min(sigma * 1.15, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma = max(sigma * 0.85, sigma_min)
                    if sigma <= sigma_min * 1.1:
                        # Stuck at very small sigma, break to save time for other restarts or polish
                        break

        # Multi-start strategy
        # We divide the budget among 3 distinct phases/starts.
        # Start 1: Global search from center with large sigma.
        # Start 2: Global search from a random point.
        # Start 3: Local refinement from the best found so far.

        # Phase 1: 50% of budget
        budget1 = int(budget * 0.5)
        x0_1 = np.zeros(dim)
        run_es(x0_1, 2.0, budget1, deadline)

        # Phase 2: 30% of budget
        budget2 = int(budget * 0.3)
        if total_calls < budget and time.time() < deadline:
            x0_2 = rng.uniform(-3, 3, dim)
            run_es(x0_2, 1.5, budget2, deadline)

        # Phase 3: Remaining budget for local refinement
        if best_x is not None and total_calls < budget and time.time() < deadline:
            x0_3 = best_x.copy()
            # Start with a smaller sigma for local refinement
            run_es(x0_3, 0.5, budget - total_calls, deadline)
            
            # Final polish: (1+1) ES with very small steps
            if total_calls < budget and time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = 0.05
                while total_calls < budget and time.time() < deadline:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.1, 0.2)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma = max(sigma * 0.9, 1e-6)
                        if sigma <= 1e-6:
                            break

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
