"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard.
    
    This implementation refines the restarting ES family by removing the 
    detrimental hard-restart (sigma=2.0 reset) and L-BFGS polish, which 
    were observed to lower the AOCC score compared to a pure, uninterrupted 
    (1+1)-ES run. It retains the time-guard structure for safety.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initialize
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_made = 1
        
        sigma = 2.0
        x = x_best.copy()
        
        # Run (1+1)-ES for the remaining budget
        # We use the full budget for the ES, as observed that 
        # uninterrupted ES outperforms ES+Polish for this task.
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < f_best:
                # Improvement
                f_best = fy
                x_best = y
                x = y
                sigma *= 1.5
            else:
                # No improvement
                sigma *= 0.87
                
            # Optional: Gentle restart if stuck in precision trap
            # Only trigger if sigma is extremely small and we've had many failures
            # This is a safeguard, but the 1/5 rule usually handles it.
            # If sigma becomes too small, we might be at the optimum.
            # We do NOT reset x, only nudge sigma up slightly if needed?
            # Actually, standard 1/5 rule is sufficient. 
            # If sigma < 1e-10, it's effectively 0. 
            # We can leave it as is.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
