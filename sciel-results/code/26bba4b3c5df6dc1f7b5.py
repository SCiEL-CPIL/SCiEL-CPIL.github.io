"""Black-box continuous optimizer for BBOB-style Rastrigin (f3)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with success-rate adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin

        rng = np.random.default_rng(42)
        global_best_x = np.full(dim, np.nan)
        global_best_f = np.inf
        total_evals = 0

        # Parameters for (1+1)-ES
        # Initial step size
        sigma = 2.0
        # Success rate target
        target_success = 0.2
        # Step size adaptation parameters
        chi_n = 1.0 # Simplified, using fixed factor for stability in 5D
        c_sigma = 1.0 / (dim + 4)
        d_sigma = 1.0
        
        def es_search(x0, sigma0, evals_left, func, global_best_x, global_best_f):
            x = x0.copy()
            fx = func(x)
            evals_used = 1
            best_x = x.copy()
            best_f = fx
            sigma = sigma0
            
            # Success rate tracking
            successes = 0
            window = 10 # Evaluate success rate over last 10 steps
            
            while evals_used < evals_left and time.time() < deadline:
                # Generate candidate
                z = rng.standard_normal(dim)
                y = x + sigma * z
                fy = func(y)
                evals_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    pass # No update to x

                # Adapt sigma based on success rate
                # Simple adaptation: if success rate is too high, increase sigma; else decrease
                # Using a simple heuristic for (1+1)-ES
                # More robust: track success rate over a window
                # For simplicity and speed, we use a direct update based on the last step
                # But to be more stable, let's use a simple memory
                # Here we use a simple multiplicative update
                # If we improved, increase sigma slightly, else decrease
                # This is a basic (1+1)-ES adaptation
                
                # A more standard approach is to use the success rate over a window
                # Let's implement a simple sliding window success rate
                # But for speed, we'll use a simplified version:
                # Update sigma every 10 steps based on the success count in that window
                if evals_used % window == 0:
                    success_rate = successes / window
                    successes = 0
                    # Adapt sigma
                    # If success rate > target, increase sigma
                    # If success rate < target, decrease sigma
                    # Use a geometric mean adjustment
                    sigma *= np.exp(c_sigma * (success_rate - target_success))
                    sigma = max(1e-5, min(sigma, 10.0))

            return best_x, best_f, evals_used, global_best_x, global_best_f

        # Multi-start strategy
        # We divide the budget into several restarts
        # Each restart gets a portion of the budget
        # The first restart gets a larger portion to find a good basin
        # Subsequent restarts start from random points or the best found so far
        
        num_restarts = 10
        # Allocate budget: first restart gets 30%, others get ~7% each
        # Total 30 + 9*7 = 93%, leaving 7% for final polishing
        allocations = []
        remaining_budget = budget
        for i in range(num_restarts):
            if i == 0:
                alloc = int(budget * 0.3)
            else:
                alloc = int(budget * 0.07)
            allocations.append(alloc)
        
        # Adjust the last allocation to use any remaining budget
        total_allocated = sum(allocations)
        if total_allocated < budget:
            allocations[-1] += (budget - total_allocated)
        
        # Start points
        # First restart: start at origin or random?
        # Rastrigin has global min at 0. Starting at 0 is a good idea.
        start_points = []
        start_points.append(np.zeros(dim))
        
        for i in range(1, num_restarts):
            # Random start
            start_points.append(rng.uniform(-5, 5, dim))
        
        for i, alloc in enumerate(allocations):
            if total_evals >= budget or time.time() > deadline:
                break
            
            start_x = start_points[i]
            # If we have a global best, we might want to restart from there sometimes?
            # No, multi-start is about diversity.
            # However, if the global best is very good, we can refine it.
            # Let's just run ES from the start point.
            
            # Initial sigma for this restart
            # If this is a refinement of the best, use smaller sigma
            # If this is a new exploration, use larger sigma
            if i == 0:
                sigma0 = 2.0
            else:
                # Random restarts can use a medium sigma
                sigma0 = 1.5
            
            bx, bf, evals_used, gb_x, gb_f = es_search(
                start_x, sigma0, alloc, func, global_best_x, global_best_f)
            
            total_evals += evals_used
            global_best_x = gb_x
            global_best_f = gb_f
            
            # If we found a very good solution, we might want to do a final polish
            # But the loop handles the budget.

        # Final polish: if we have budget left, do a small ES from the global best
        remaining = budget - total_evals
        if remaining > 20 and not np.isnan(global_best_x[0]):
            bx, bf, evals_used, gb_x, gb_f = es_search(
                global_best_x, 0.5, remaining, func, global_best_x, global_best_f)
            total_evals += evals_used
            global_best_x = gb_x
            global_best_f = gb_f

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
