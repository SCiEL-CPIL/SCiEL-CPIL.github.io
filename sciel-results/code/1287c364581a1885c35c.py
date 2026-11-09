"""Optimizer for BBOB f23 (Katsuura) using Multi-Start (1+1)-ES with LHS initialization and success-rate control."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with LHS initialization and success-rate-based sigma control for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin well within 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # Use 40 points for initial spread to leave more budget for local search
        n_lhs = 40
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                return x_global_best, fx_global_best
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best

        # Phase 2: Multi-Start Local Search with (1+1)-ES
        # Use success-rate-based sigma adaptation (target 1/3 success rate)
        # This is more robust than fixed increase/decrease factors
        
        # We'll do multiple restarts, each with a local (1+1)-ES
        # Each restart gets a portion of the remaining budget
        
        # Strategy: 
        # - First few restarts: refine the best LHS points
        # - Later restarts: jump from global best to explore new basins
        # - Final restart: long run from global best for final convergence
        
        # Determine number of restarts and budget per restart
        # We want about 60-80 steps per restart for good convergence
        # But leave some budget for a final long polish
        
        # Let's use adaptive restart lengths
        # Start with shorter restarts, increase as we converge
        
        restart_configs = []
        remaining = remaining_budget
        
        # First, refine the top few LHS points (we don't have sorted, so just use global best)
        # Then do exploration restarts, then a final polish
        
        # Phase 2a: Short refinement of global best (30 steps)
        if remaining >= 30:
            restart_configs.append(('refine_best', 30))
            remaining -= 30
        
        # Phase 2b: Exploration restarts with jumps
        # Do several medium-length restarts with jumps from global best
        n_explore = 4
        explore_len = max(20, remaining // (n_explore + 1))
        for i in range(n_explore):
            if remaining >= explore_len:
                restart_configs.append(('explore', explore_len))
                remaining -= explore_len
        
        # Phase 2c: Final long polish from global best
        if remaining >= 10:
            restart_configs.append(('polish', remaining))
            remaining = 0
        
        # If we have leftover, add to polish
        if remaining > 0:
            if restart_configs:
                # Add to the last (polish) restart
                if restart_configs[-1][0] == 'polish':
                    restart_configs[-1] = ('polish', restart_configs[-1][1] + remaining)
                else:
                    restart_configs.append(('polish', remaining))
            else:
                restart_configs.append(('polish', remaining))
        
        sigma_init = 1.5
        min_sigma = 0.001
        max_sigma = 5.0
        jump_scale = 2.5
        target_success_rate = 1.0 / 3.0
        window_size = 15  # Window for success rate tracking
        
        for trial_type, trial_length in restart_configs:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                return x_global_best, fx_global_best
            
            # Determine starting point
            if trial_type == 'refine_best':
                x = x_global_best.copy()
            elif trial_type == 'explore':
                # Jump from global best
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            else:  # polish
                x = x_global_best.copy()
            
            # Evaluate starting point
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            successes = 0
            total_steps = 0
            
            # Run local search for up to trial_length steps
            for step in range(trial_length - 1):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    update_best(x, fx)
                else:
                    pass
                
                # Adapt sigma based on success rate over the window
                if total_steps >= window_size:
                    success_rate = successes / total_steps
                    if success_rate > target_success_rate:
                        sigma = min(sigma * 1.2, max_sigma)
                    elif success_rate < target_success_rate:
                        sigma = max(sigma * 0.8, min_sigma)
                    # Reset window
                    successes = 0
                    total_steps = 0
                    
        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
