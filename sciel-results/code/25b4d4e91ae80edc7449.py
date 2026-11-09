"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Basin-Hopping with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # Best solution tracking
        best_x = rng.uniform(-5, 5, self.dim)
        best_f = func(best_x)
        
        # Budget tracking
        calls_made = 1
        max_calls = self.budget
        
        # Strategy: Multi-start with Basin Hopping
        # 1. Randomly initialize a point
        # 2. Perturb it (large step for Rastrigin's periodicity)
        # 3. Refine with a small (1+1)-ES or gradient-free local search
        # 4. Repeat until budget or time runs out
        
        # Parameters
        hop_sigma = 2.0  # Large step to escape local minima
        local_sigma = 0.5 # Small step for refinement
        local_iterations = 20 # Refinement steps per hop
        
        while calls_made < max_calls:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate a candidate for the next hop
            # We can either jump from the current best or from a random point
            # To ensure global exploration, mix random restarts with local search around best
            if rng.random() < 0.3:
                # 30% chance: Full random restart
                x = rng.uniform(-5, 5, self.dim)
            else:
                # 70% chance: Jump from current best
                x = np.clip(best_x + hop_sigma * rng.standard_normal(self.dim), -5, 5)
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_x = x
                best_f = fx
            
            # Local Refinement (Mini (1+1)-ES)
            # We do this on the current candidate x to see if we can improve locally
            # without losing the global exploration benefit of the hop
            local_x = x.copy()
            local_f = fx
            
            for _ in range(local_iterations):
                if calls_made >= max_calls:
                    break
                if time.time() > deadline:
                    break
                
                step = local_sigma * rng.standard_normal(self.dim)
                y = np.clip(local_x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < local_f:
                    local_x = y
                    local_f = fy
                    local_sigma *= 1.1 # Increase step slightly if improving
                else:
                    local_sigma *= 0.9 # Decrease step if failing
                    if local_sigma < 0.01:
                        break
            
            if local_f < best_f:
                best_x = local_x
                best_f = local_f

        return best_x, best_f
# EVOLVE-BLOCK-END
