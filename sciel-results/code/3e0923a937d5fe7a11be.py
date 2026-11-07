"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1) Evolution Strategy for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        deadline = time.time() + 290.0
        d = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        calls_used = 0
        
        # Parameters for (1+1)-ES
        # Start with a relatively large sigma to explore the [-5, 5] box
        sigma = 2.0 
        # Restart if no improvement for this many steps
        stagnation_limit = 20
        # Minimum sigma
        sigma_min = 1e-4
        
        # We will use a loop that runs until budget or time is exhausted.
        # Each "cycle" is a (1+1)-ES run that restarts if stagnated.
        
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize current candidate
            # Start with a random point in the box
            curr_x = rng.uniform(self.bounds[0], self.bounds[1], d)
            curr_f = func(curr_x)
            calls_used += 1
            
            if curr_f < best_f:
                best_f = curr_f
                best_x = curr_x.copy()
            
            # Local search phase
            no_improve_count = 0
            local_sigma = sigma
            
            while calls_used < budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                # Use Gaussian perturbation
                delta = rng.normal(0, local_sigma, d)
                cand_x = curr_x + delta
                cand_x = np.clip(cand_x, self.bounds[0], self.bounds[1])
                
                cand_f = func(cand_x)
                calls_used += 1
                
                if cand_f < curr_f:
                    # Improvement
                    curr_x = cand_x
                    curr_f = cand_f
                    no_improve_count = 0
                    # Increase sigma slightly to allow larger steps? 
                    # Standard 1/5 rule or similar. Let's just keep it or increase slightly.
                    # For Rastrigin, we might want to stay in the valley.
                    local_sigma = min(local_sigma * 1.1, 2.0) 
                    
                    if curr_f < best_f:
                        best_f = curr_f
                        best_x = curr_x.copy()
                else:
                    # No improvement
                    no_improve_count += 1
                    # Decrease sigma
                    local_sigma = max(local_sigma * 0.8, sigma_min)
                
                # Check for stagnation
                if no_improve_count >= stagnation_limit:
                    break
            
            # If we broke out of inner loop due to stagnation, we restart the outer loop
            # which picks a new starting point (random).
            # If we broke out due to time/budget, we exit.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
