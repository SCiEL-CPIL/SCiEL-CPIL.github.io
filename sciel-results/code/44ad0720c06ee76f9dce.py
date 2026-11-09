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
                    
                # Adaptive sigma with 1/5 success rule (CMA-ES inspired)
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
                    # More aggressive adaptation: aim for 1/5 success rate
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.75
                    else:
                        sigma *= 1.02  # slight increase if close to target
                    success_window = success_window[-window_size//2:]
                
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # If sigma is very small, try a larger step to escape potential local optima
                if sigma < 1e-4 and rng.random() < 0.1:
                    sigma = 0.1

        # Strategy: Multiple restarts with different initializations and sigma
        # 1. Start with a focused random search to find a good region
        # 2. Then refine with (1+1)-ES using adaptive sigma
        # 3. Final polish phase for high-precision convergence

        # Phase 1: Focused exploration (5% of budget)
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

        # Phase 2a: Primary refinement from best point (40% of remaining)
        if best_x is not None and remaining_budget > 10:
            primary_budget = min(remaining_budget, max(10, int(remaining_budget * 0.4)))
            es_run(rng, best_x, 1.0, primary_budget)
            remaining_budget = total_budget - calls_used
            
        # Phase 2b: Secondary refinement from a nearby random point (30% of remaining)
        if remaining_budget > 10:
            secondary_budget = min(remaining_budget, max(10, int(remaining_budget * 0.3)))
            # Start near the best point with some perturbation
            x2 = best_x + rng.uniform(-1.0, 1.0, dim)
            x2 = np.clip(x2, -5, 5)
            es_run(rng, x2, 0.8, secondary_budget)
            remaining_budget = total_budget - calls_used
            
        # Phase 2c: Tertiary refinement from a fresh random point (20% of remaining)
        if remaining_budget > 10:
            tertiary_budget = min(remaining_budget, max(10, int(remaining_budget * 0.2)))
            x3 = rng.uniform(-5, 5, dim)
            es_run(rng, x3, 1.5, tertiary_budget)
            remaining_budget = total_budget - calls_used
            
        # Phase 3: Final polish - high-precision local search from the best point
        if remaining_budget > 5 and best_x is not None:
            # Use a very small sigma for fine-tuning
            polish_budget = remaining_budget
            es_run(rng, best_x, 0.05, polish_budget)
            remaining_budget = total_budget - calls_used
            
        # Ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
