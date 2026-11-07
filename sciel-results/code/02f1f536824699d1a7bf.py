import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # Generate all vertices of the box [-5, 5]^dim
        # For dim=5, this is 2^5 = 32 points
        # Using product for clarity
        coords = np.array(list(itertools.product([-5.0, 5.0], repeat=self.dim)))
        
        best_x = None
        best_f = np.inf
        
        # Evaluate each vertex
        for x in coords:
            if time.time() > deadline:
                break
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except Exception:
                continue

        # If we haven't found a valid point (shouldn't happen), fallback
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        # The budget is 500, we used 32. 
        # We can use the remaining budget to do a fine-grained search around the best vertex 
        # just in case the function isn't perfectly linear or has noise, 
        # but for "Linear slope", the vertex IS the optimum.
        # However, to be robust against any slight non-linearity or if the "linear slope" 
        # is actually a very steep gradient that makes floating point issues matter, 
        # we can perform a small local search from the best vertex.
        
        # Let's add a small local polish using Nelder-Mead or similar 
        # but constrained to the box? 
        # Actually, if it's linear, moving away from the vertex increases the value. 
        # So the vertex is the local and global min. 
        # No polish needed.
        
        return best_x, best_f
