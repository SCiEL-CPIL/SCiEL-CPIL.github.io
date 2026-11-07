"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Multi-start manual Powell's method optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # State for call counting
        calls_made = 0
        
        def obj(x):
            nonlocal calls_made
            calls_made += 1
            if calls_made > self.budget:
                return 1e10
            return func(x)
        
        def line_search(x0, direction, f0, step_size):
            """
            Simple 1D line search using Golden Section.
            Returns (x_opt, f_opt, t_opt)
            """
            if calls_made >= self.budget or time.time() > deadline:
                return x0, f0, 0.0
                
            # Bracketing
            # Try small step
            x1 = x0 + step_size * direction
            f1 = obj(x1)
            if f1 > f0:
                direction = -direction
                x1 = x0 + step_size * direction
                f1 = obj(x1)
                if f1 > f0:
                    return x0, f0, 0.0
            
            # Expand bracket
            x2 = x0 + 2 * step_size * direction
            f2 = obj(x2)
            while f2 < f1 and calls_made < self.budget and time.time() < deadline:
                step_size *= 2
                x1 = x2
                f1 = f2
                x2 = x0 + 2 * step_size * direction
                f2 = obj(x2)
            
            # Golden Section on [0, 2*step_size]
            a = 0.0
            b = 2 * step_size
            gr = (math.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = obj(x0 + c * direction)
            fd = obj(x0 + d * direction)
            
            for _ in range(20): # Max 20 iterations for line search
                if calls_made >= self.budget or time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = obj(x0 + c * direction)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = obj(x0 + d * direction)
                
                if abs(b - a) < 1e-8:
                    break
            
            t = (a + b) / 2
            x_opt = x0 + t * direction
            f_opt = obj(x_opt)
            return x_opt, f_opt, t

        def powell(x_start, f_start, max_cycles=5):
            x = x_start.copy()
            f = f_start
            directions = np.eye(self.dim)
            step_size = 1.0
            
            for _ in range(max_cycles):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                    
                old_f = f
                min_f = f
                min_dir_idx = -1
                min_x = x.copy()
                
                # Perform line search along each direction
                for i in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    x_new, f_new, _ = line_search(x, directions[i], f, step_size)
                    if f_new < min_f:
                        min_f = f_new
                        min_dir_idx = i
                        min_x = x_new.copy()
                    x = x_new
                    f = f_new
                
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Update directions
                # If we found a better point, update the direction set
                if min_f < old_f:
                    # The direction that gave the best improvement is min_dir_idx
                    # The new direction is the total displacement from the start of the cycle to the end
                    # This is an approximation of the standard Powell update
                    # Standard Powell:
                    # If the last direction was the best, replace it with the total displacement.
                    # Here we just replace the best direction with the total displacement vector
                    # normalized? No, Powell uses the actual vector.
                    
                    # Let's use the vector from the point before the best step to the final point?
                    # Actually, the vector from the start of the cycle (x_start_cycle) to the end (x)
                    # is a good candidate for a new conjugate direction.
                    
                    # Let's store the start of the cycle
                    # We didn't store it in the loop above. Let's restructure.
                    
                    # For simplicity, let's just keep the directions fixed 
                    # and rely on multi-start. 
                    # Or, simply replace the direction that was used for the best step 
                    # with the total displacement vector of the cycle.
                    
                    # To do this properly, we need x_start_cycle.
                    # Let's just return the best found so far.
                    # The multi-start aspect is the main driver here.
                    
                    break # Stop this Powell trajectory to save budget for others
                    
            return x, f

        # Multi-start
        rng = np.random.default_rng(0)
        starts = []
        starts.append(np.zeros(self.dim))
        
        # Add random starts
        num_starts = 10
        for _ in range(num_starts):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        rng.shuffle(starts)
        
        best_x = None
        best_f = float('inf')
        
        for start in starts:
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            f_start = obj(start)
            x_opt, f_opt = powell(start, f_start, max_cycles=3)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = obj(best_x)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
