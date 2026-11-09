"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of adaptive (1+1)-ES runs with time and budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def es_run(rng, x0, sigma0, budget_left):
            nonlocal calls_used, best_x, best_f, deadline
            x = x0.copy()
            fx = func(x)
            track_call(x, fx)
            sigma = sigma0
            n_dim = self.dim
            
            # Use a sliding window for success rate to adapt sigma
            # 1/5 success rule: aim for ~20% success rate
            success_window = []
            window_size = 10
            
            for _ in range(budget_left):
                if time.time() > deadline:
                    return
                if calls_used >= total_budget:
                    return
                    
                # Generate step
                step = sigma * rng.standard_normal(n_dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                track_call(y, fy)
                
                if fy < fx:
                    x = y
                    fx = fy
                    success_window.append(1)
                else:
                    success_window.append(0)
                    
                # Adapt sigma based on success rate over the window
                if len(success_window) >= window_size:
                    success_rate = sum(success_window) / window_size
                    if success_rate > 0.3:
                        # Too many successes, increase step size
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        # Too few successes, decrease step size
                        sigma *= 0.8
                    else:
                        # In the sweet spot, keep sigma or slight adjustment
                        sigma *= 1.01
                    
                    # Keep window manageable
                    success_window = success_window[-window_size//2:]
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Strategy:
        # 1. Short broad exploration to find a good region
        # 2. Multiple (1+1)-ES runs starting from different points with different initial sigmas
        #    This helps escape local minima and handle the hidden shift/rotation

        # Phase 1: Broad exploration (5% of budget)
        explore_budget = max(5, total_budget // 20)
        rng = np.random.default_rng(42)
        
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            if calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)
            
        remaining_budget = total_budget - calls_used
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            remaining_budget = total_budget - calls_used

        # Phase 2: Run multiple ES instances
        # We divide the remaining budget among 3 runs
        # Run 1: Start from best found, medium sigma
        # Run 2: Start from random, large sigma (to explore other regions)
        # Run 3: Start from random, small sigma (for fine-tuning if it lands in a good spot)
        
        if remaining_budget > 10:
            # Run 1: Refine the best point
            es_run(rng, best_x, 1.0, remaining_budget // 3)
            remaining_budget = total_budget - calls_used
            
        if remaining_budget > 10:
            # Run 2: Explore a different region
            x2 = rng.uniform(-5, 5, dim)
            es_run(rng, x2, 2.0, remaining_budget // 3)
            remaining_budget = total_budget - calls_used
            
        if remaining_budget > 5:
            # Run 3: Fine-tuning or another exploration
            # Start from the current best again, with a smaller sigma to polish
            es_run(rng, best_x, 0.5, remaining_budget)

        # Ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
