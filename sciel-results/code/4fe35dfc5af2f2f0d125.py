"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restarting (1+1)-ES with stagnation detection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng()
        
        # Global best tracking
        best_x = rng.uniform(-5, 5, self.dim)
        best_f = func(best_x)
        calls_used = 1
        
        # Parameters for restart strategy
        # We divide the budget into chunks or restart based on stagnation
        # Let's use a stagnation-based restart.
        # If no improvement for 'stagnation_limit' steps, restart.
        stagnation_limit = 20
        min_steps_per_run = 10
        
        while calls_used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize a new run
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            
            # Update global best if this new start is better
            if fx < best_f:
                best_x = x.copy()
                best_f = fx
            
            sigma = 1.0
            no_improve_count = 0
            
            # Run ES until stagnation or budget/time exhausted
            while calls_used < self.budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve_count = 0
                    # Adapt sigma: increase on success
                    sigma *= 1.2
                    # Update global best
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    no_improve_count += 1
                    # Adapt sigma: decrease on failure
                    sigma *= 0.85
                    
                # Check for stagnation to trigger restart
                if no_improve_count >= stagnation_limit:
                    break
                
                # Ensure we don't run forever if sigma becomes tiny
                if sigma < 1e-6:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
