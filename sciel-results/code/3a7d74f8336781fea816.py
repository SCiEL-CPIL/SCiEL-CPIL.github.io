"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start stochastic local search with adaptive local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def local_search(x_start, f_start, max_local_calls):
            nonlocal calls_made
            x = x_start.copy()
            fx = f_start
            sigma = 0.5  # Initial step size for local polish
            
            # Local search loop
            local_calls = 0
            while local_calls < max_local_calls and calls_made < budget:
                # Time check
                if time.time() > deadline:
                    break
                
                # Generate candidate
                direction = rng.standard_normal(n)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls_made += 1
                local_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fy)
                    # Adapt sigma: increase slightly if successful, but keep it small for polishing
                    sigma *= 1.1
                    sigma = min(sigma, 2.0)
                else:
                    # Decrease sigma if stuck, to refine
                    sigma *= 0.8
                    sigma = max(sigma, 1e-3)
                    
            return x, fx

        # Phase 1: Coarse Exploration
        # We divide the budget: ~30% for random starts, 70% for local polishing
        # But we interleave to ensure early AOCC points are decent.
        # Strategy: Generate a random point, evaluate, update best, then do a short local polish.
        # Repeat.
        
        # To maximize AOCC, we want best_f to drop quickly.
        # Random uniform points in [-5,5]^5 are often far from the optimum of Rosenbrock.
        # However, local search from a good random start can be very effective.
        
        # Let's try a "Best-First" approach with multiple restarts.
        # 1. Evaluate center (0,0,0,0,0) as a safe start.
        x0 = np.zeros(n)
        f0 = func(x0)
        calls_made += 1
        update_best(x0, f0)
        
        # 2. Loop: Pick a new random start, or refine current best?
        # Given Rosenbrock's landscape, random starts are risky.
        # Better: Start from best, do local search. If stuck, random jump.
        
        # Hybrid: 
        # - Keep a "current" point.
        # - Do local search from "current".
        # - If local search yields no improvement for a while, do a random jump.
        
        current_x = x0.copy()
        current_f = f0
        
        # Parameters
        local_budget_per_iter = 20  # Calls per local polish attempt
        random_jump_threshold = 100 # If no global improvement in this many calls, jump
        
        calls_since_global_improve = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Determine how many calls we can afford for local search
            remaining = budget - calls_made
            local_calls_to_make = min(local_budget_per_iter, remaining)
            
            if local_calls_to_make <= 0:
                break
                
            # Perform local search from current_x
            new_x, new_f = local_search(current_x, current_f, local_calls_to_make)
            
            # Check if we improved globally
            if new_f < best_f:
                calls_since_global_improve = 0
            else:
                calls_since_global_improve += local_calls_to_make
                
            current_x = new_x
            current_f = new_f
            
            # If stuck for a long time, do a random restart
            if calls_since_global_improve > random_jump_threshold:
                # Random restart
                x_new = rng.uniform(-5, 5, n)
                f_new = func(x_new)
                calls_made += 1
                update_best(x_new, f_new)
                current_x = x_new
                current_f = f_new
                calls_since_global_improve = 0
                # Reset sigma implicitly by starting local_search fresh with default sigma
                # local_search initializes sigma=0.5 each time it's called? 
                # No, local_search takes x_start and f_start, but sigma is local to the function.
                # We should pass a sigma or let it reset.
                # In the implementation above, sigma is initialized inside local_search.
                # So it resets to 0.5 every time. This is good for restarts.
                
        return best_x, best_f
# EVOLVE-BLOCK-END
