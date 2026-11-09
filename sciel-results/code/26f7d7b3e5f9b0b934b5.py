"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of diverse (1+1)-ES runs with time and budget management."""

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
            success_window = []
            window_size = 15
            
            for _ in range(budget_left):
                if time.time() > deadline:
                    return
                if calls_used >= total_budget:
                    return
                    
                # 1/5 success rule for sigma adaptation
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
                    
                if len(success_window) >= window_size:
                    success_rate = sum(success_window) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    success_window = success_window[-window_size//2:]
                
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Strategy: Multiple restarts with different initializations and sigma
        # 1. Start with a broad random search to find a good region
        # 2. Then refine with (1+1)-ES

        # Phase 1: Broad exploration (10% of budget)
        explore_budget = max(10, total_budget // 10)
        rng = np.random.default_rng(42)
        
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            if calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)
            
        # Phase 2: Refinement with multiple ES runs
        # Use the best point found so far as a starting point for one run
        # And use other random points for additional runs
        
        remaining_budget = total_budget - calls_used
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            remaining_budget = total_budget - calls_used

        if best_x is not None and remaining_budget > 10:
            # Run 1: Refine from the best found point
            es_run(rng, best_x, 1.0, min(remaining_budget, total_budget // 2))
            remaining_budget = total_budget - calls_used
            
        if remaining_budget > 10:
            # Run 2: New random start with medium sigma
            x2 = rng.uniform(-5, 5, dim)
            es_run(rng, x2, 2.0, min(remaining_budget, total_budget // 3))
            remaining_budget = total_budget - calls_used
            
        if remaining_budget > 10:
            # Run 3: New random start with small sigma for fine-tuning
            x3 = rng.uniform(-5, 5, dim)
            es_run(rng, x3, 0.5, remaining_budget)
            
        # Ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
