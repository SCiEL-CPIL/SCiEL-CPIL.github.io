"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Coordinate Descent with Adaptive Step Sizes.
    
    Strategy:
    1. Start with a random point.
    2. For each dimension, try to move in + and - directions.
    3. If a move improves the function value, accept it and increase the step size for that dimension.
    4. If no move improves, decrease the step size.
    5. Repeat until budget is exhausted or no improvement is found for several full passes.
    
    This is structurally different from (1+1)-ES because it is deterministic in its direction search 
    (axis-aligned) and uses a separate step size per dimension, allowing it to "slide" along valleys 
    more effectively than isotropic Gaussian noise.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Deterministic initial point to ensure reproducibility if needed, 
        # but random is fine too. Let's use a fixed seed for consistency.
        rng = np.random.default_rng(42)
        
        # Initialize x
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        # Initialize step sizes per dimension
        # Start with a moderate step size
        steps = np.ones(dim) * 1.0
        
        # Track best
        best_x = x.copy()
        best_fx = fx
        
        # Variables for adaptive logic
        no_improve_count = 0
        max_no_improve = 10  # If we do 10 full passes without any improvement, we stop or restart
        
        for iteration in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break
            
            improved_in_pass = False
            
            # Cycle through dimensions
            for d in range(dim):
                # Try positive direction
                x_new = x.copy()
                x_new[d] += steps[d]
                x_new = np.clip(x_new, -5, 5)
                
                # Only evaluate if we haven't used all budget
                if time.time() > deadline:
                    break
                
                fx_new = func(x_new)
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    steps[d] *= 1.5  # Increase step size
                    improved_in_pass = True
                    
                    # Update global best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    # Try negative direction
                    x_new = x.copy()
                    x_new[d] -= steps[d]
                    x_new = np.clip(x_new, -5, 5)
                    
                    if time.time() > deadline:
                        break
                    
                    fx_new = func(x_new)
                    
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        steps[d] *= 1.5  # Increase step size
                        improved_in_pass = True
                        
                        # Update global best
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        # No improvement in either direction for this dimension
                        steps[d] *= 0.5  # Decrease step size
                        
                        # If step size becomes too small, it's effectively converged for this dim
                        if steps[d] < 1e-6:
                            steps[d] = 1e-6
            
            if time.time() > deadline:
                break
            
            if improved_in_pass:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # If we are stuck, try a random restart or a large random jump
                # This helps escape local minima in rotated landscapes
                if no_improve_count >= max_no_improve:
                    # Random restart around current best or global random
                    # Let's try a local perturbation of the best found so far
                    if rng.random() < 0.5:
                        x = best_x + rng.normal(0, 0.5, dim)
                    else:
                        x = rng.uniform(-5, 5, dim)
                    
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    
                    # Reset step sizes to a medium value
                    steps = np.ones(dim) * 1.0
                    no_improve_count = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
