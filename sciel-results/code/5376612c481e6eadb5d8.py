"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate Descent with Golden Section Search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # Initial point: random in [-5, 5]
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        # Golden Section Search parameters
        # We use a fixed number of iterations per coordinate per pass
        # Bent Cigar is smooth, so 20-30 iterations of GS gives high precision
        gs_iters = 30 
        # Total budget is 500 calls.
        # Each full pass over D dimensions costs D * gs_iters calls.
        # For D=5, one pass costs 150 calls.
        # We can afford about 3 full passes (450 calls) and have some left.
        # Let's calculate max passes based on budget.
        calls_per_pass = self.dim * gs_iters
        max_passes = max(1, (self.budget - 10) // calls_per_pass)
        
        # If budget is large, we can do more passes or increase precision.
        # However, for Bent Cigar, 30 GS iterations usually converges to machine precision
        # relative to the scale.
        
        for _ in range(max_passes):
            # Check time and budget
            if time.time() > deadline:
                break
            
            for i in range(self.dim):
                if time.time() > deadline:
                    break
                
                # Optimize dimension i
                # We search in the range [-5, 5] for x[i]
                # To be more efficient, we could narrow the range based on previous best,
                # but since the domain is fixed [-5,5] and the function is convex,
                # searching the whole box is safe.
                # However, a tighter initial bracket might help. 
                # For now, use full box [-5, 5].
                
                best_x_i = x[i]
                best_f_i = fx
                
                # Golden Section Search
                a = -5.0
                b = 5.0
                phi = (1.0 - np.sqrt(5.0)) / 2.0  # ~ -0.618, but we use standard GS constants
                # Standard GS:
                # c = b - phi * (b - a)
                # d = a + phi * (b - a)
                # Actually, let's use the standard ratio r = (sqrt(5)-1)/2 ~ 0.618
                r = (np.sqrt(5.0) - 1.0) / 2.0
                
                c = a + (1.0 - r) * (b - a)
                d = a + r * (b - a)
                
                # Evaluate f at c and d
                # We need to construct the vector x with x[i] replaced by c or d
                x_c = x.copy()
                x_c[i] = c
                f_c = func(x_c)
                
                x_d = x.copy()
                x_d[i] = d
                f_d = func(x_d)
                
                # Keep track of best found in this 1D search
                if f_c < best_f_i:
                    best_f_i = f_c
                    best_x_i = c
                if f_d < best_f_i:
                    best_f_i = f_d
                    best_x_i = d
                    
                for _ in range(gs_iters - 2): # We already evaluated 2 points
                    if time.time() > deadline:
                        break
                    
                    if f_c < f_d:
                        # Minimum is in [a, d]
                        b = d
                        d = c
                        f_d = f_c
                        c = a + (1.0 - r) * (b - a)
                        x_c = x.copy()
                        x_c[i] = c
                        f_c = func(x_c)
                        if f_c < best_f_i:
                            best_f_i = f_c
                            best_x_i = c
                    else:
                        # Minimum is in [c, b]
                        a = c
                        c = d
                        f_c = f_d
                        d = a + r * (b - a)
                        x_d = x.copy()
                        x_d[i] = d
                        f_d = func(x_d)
                        if f_d < best_f_i:
                            best_f_i = f_d
                            best_x_i = d

                # Update x and fx with the best found in this dimension
                x[i] = best_x_i
                fx = best_f_i
                
        return x, fx
# EVOLVE-BLOCK-END
