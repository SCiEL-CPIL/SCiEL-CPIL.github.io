"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restarting (1+1)-ES with adaptive step-size, stagnation restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-4, 4, self.dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Parameters for restart strategy
        max_restarts = 20
        stagnation_threshold = 30
        restart_count = 0
        
        while calls_made < self.budget and restart_count < max_restarts:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Initialize current run state
            x = x_best.copy()
            # Start with a moderate sigma
            sigma = 1.5 
            stagnation_count = 0
            local_best_val = f_best
            local_best_x = x_best.copy()
            
            # Run (1+1)-ES for this restart phase
            while calls_made < self.budget:
                if time.time() > deadline:
                    break
                
                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < local_best_val:
                    # Improvement
                    local_best_val = fy
                    local_best_x = y
                    x = y
                    sigma *= 1.3  # More aggressive expansion on success
                    stagnation_count = 0
                    
                    # Update global best
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                else:
                    # No improvement
                    sigma *= 0.85  # Faster contraction on failure
                    stagnation_count += 1
                
                # Check for stagnation to trigger restart
                if stagnation_count >= stagnation_threshold:
                    break
            
            restart_count += 1
            
            # If we broke out due to budget/time, the outer loop condition handles it.
            # If sigma becomes too small, restart even if not fully stagnant
            if sigma < 1e-7 and calls_made < self.budget:
                restart_count += 1
        
        # Final polish: try to improve with very small steps around x_best
        if calls_made < self.budget:
            polish_sigma = 0.1
            for _ in range(min(50, self.budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(x_best + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < f_best:
                    f_best = fy
                    x_best = y
        
        return x_best, f_best
# EVOLVE-BLOCK-END
