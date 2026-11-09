"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer with 1/5 success rule for adaptive step size.
    Uses a single full-budget run to exploit the anisotropic landscape of the Bent Cigar.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initialize
        rng = np.random.default_rng(0)
        x = rng.uniform(-4, 4, dim)  # Start slightly inside bounds to avoid clipping issues
        fx = func(x)
        calls_used = 1
        
        # Initial step size
        sigma = 2.0
        
        # 1/5 rule variables
        # We keep a history of the last 15 steps to check if at least 3 succeeded
        # Standard 1/5 rule: if 3 out of last 15 steps were successful, increase sigma
        # Otherwise decrease.
        # For better performance on Bent Cigar, we might use a slightly different window or factor.
        # Let's stick to a robust variant: 
        # success_count in last 15 steps >= 3 -> sigma *= 1.1
        # else -> sigma *= 0.9
        
        window_size = 15
        successes_in_window = []
        
        best_x = x.copy()
        best_fx = fx
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            
            # Update 1/5 rule
            if fy < fx:
                x = y
                fx = fy
                successes_in_window.append(1)
            else:
                successes_in_window.append(0)
            
            # Maintain window size
            if len(successes_in_window) > window_size:
                successes_in_window.pop(0)
            
            # Adapt sigma if we have enough history
            if len(successes_in_window) == window_size:
                success_count = sum(successes_in_window)
                if success_count >= 3:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            
            # Early termination if sigma becomes too small
            if sigma < 1e-6:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
