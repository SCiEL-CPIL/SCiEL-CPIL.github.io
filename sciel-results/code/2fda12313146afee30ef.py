"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB f21 (101 peaks).
    Uses a two-phase approach:
    1. Random Global Scan to locate a promising basin quickly (early AOCC).
    2. Adaptive (1+1)-ES for local exploitation. This is much faster than CMA-ES
       for 5D problems and effectively refines the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(1234)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Scan
        # For f21, random search is very effective at finding *a* peak.
        # We allocate ~20% of the budget here.
        scan_budget = int(0.2 * budget)
        
        # We can do a simple random search, but to be slightly smarter, 
        # we can sample from a few different distributions.
        # 1. Uniform
        # 2. Gaussian centered at 0
        for i in range(scan_budget):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Mix of uniform and gaussian to cover different scales
            if i % 2 == 0:
                x = rng.uniform(-5, 5, size=dim)
            else:
                x = rng.normal(0, 2.0, size=dim)
                x = np.clip(x, -5, 5)
            
            call_func(x)
        
        # If we haven't made any calls (shouldn't happen), break
        if calls_made == 0:
            x = np.zeros(dim)
            call_func(x)
            
        # Phase 2: Local Exploitation with Adaptive (1+1)-ES
        # We use the best found so far as the starting point.
        if best_x is None:
            return best_x, best_f
            
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Initial step size: small enough to stay in the basin, 
        # but large enough to explore it.
        # The peaks in f21 are somewhat narrow.
        step_size = 0.5 
        
        # Adaptation parameters
        step_size_increase = 1.05
        step_size_decrease = 0.9
        
        # Anti-stagnation: If no improvement for a while, take a big jump
        stagnation_count = 0
        max_stagnation = 20
        jump_size = 2.0

        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate trial
            # Add some noise to the step direction to handle rotation better
            direction = rng.standard_normal(dim)
            trial_x = x_curr + step_size * direction
            trial_x = np.clip(trial_x, -5, 5)
            
            trial_f = call_func(trial_x)
            
            if trial_f < f_curr:
                # Improvement
                x_curr = trial_x
                f_curr = trial_f
                step_size *= step_size_increase
                stagnation_count = 0
            else:
                # No improvement
                step_size *= step_size_decrease
                stagnation_count += 1
                
                # If stuck for a long time, perform a large random jump
                # This helps escape narrow local minima if we are not in the global one
                # or if the basin is extremely flat/noisy.
                if stagnation_count > max_stagnation:
                    jump_x = x_curr + rng.normal(0, jump_size, size=dim)
                    jump_x = np.clip(jump_x, -5, 5)
                    jump_f = call_func(jump_x)
                    
                    if jump_f < f_curr:
                        x_curr = jump_x
                        f_curr = jump_f
                        step_size = 1.0 # Reset step size
                        stagnation_count = 0
                    else:
                        # Keep shrinking or restart stagnation counter to avoid infinite jumps
                        # We just reset the counter to allow another jump later
                        stagnation_count = 0
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
