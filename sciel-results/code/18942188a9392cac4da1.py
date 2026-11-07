"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with initial random sampling and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Search
        # Sample a few points to find a good starting basin
        n_initial = 10
        if self.budget < n_initial + 10:
             n_initial = max(1, self.budget // 2)
             
        xs = rng.uniform(-5, 5, (n_initial, self.dim))
        best_x = None
        best_fx = float('inf')
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x_i = xs[i]
            f_i = func(x_i)
            if f_i < best_fx:
                best_fx = f_i
                best_x = x_i.copy()
                
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            best_fx = func(best_x)
            
        # Phase 2: Local Refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_fx
        
        sigma = 2.0
        stall_counter = 0
        stall_threshold = 40
        
        remaining_budget = self.budget - n_initial
        for i in range(remaining_budget):
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            if fy < fx:
                # Accept step
                x = y
                fx = fy
                # Increase sigma to continue exploration in this direction
                sigma *= 1.2
                stall_counter = 0
                
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Reject step
                # Decrease sigma
                sigma *= 0.8
                stall_counter += 1
                
                # If stalled for too long, reset sigma to a larger value to escape local minima
                if stall_counter >= stall_threshold:
                    sigma = 2.5
                    stall_counter = 0
                    
        return best_x, best_fx
# EVOLVE-BLOCK-END
