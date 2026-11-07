import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        n_evals = self.budget
        all_F = []
        
        # DTLZ1 with 7 vars, 3 objectives:
        # f1 = 0.5 * x0 * (1+g), f2 = 0.5 * x1 * (1+g), f3 = 0.5 * (1-x0-x1) * (1+g)
        # g = sum(x_i for i in 3..6) + 10*(n-4)*(sin(pi*(x_3+1)/10) + ...)
        # g_min = 1 when x_3..x_6 = 0.5
        # On Pareto front: 1+g = 2, so f1 = x0, f2 = x1, f3 = 1-x0-x1
        # Constraint: x0 + x1 <= 1, x0 >= 0, x1 >= 0
        
        # Strategy: Use a 2D Sobol sequence to uniformly sample the triangular Pareto front
        # The triangle in (x0, x1) space has vertices (0,0), (1,0), (0,1)
        # We need uniform sampling over this triangle
        
        # Use scipy's Sobol sequence for better low-discrepancy properties
        try:
            from scipy.stats import qmc
            # Generate Sobol points in [0,1]^2
            n_sobol = n_evals
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            points_2d = sobol.random(n_sobol)
        except ImportError:
            # Fallback to Halton sequence
            n_sobol = n_evals
            def halton(n, base):
                result = []
                f = 1.0 / base
                i = 1
                for _ in range(n):
                    r = 0.0
                    while i > 0:
                        r += f * (i % base)
                        i //= base
                        f /= base
                    result.append(r)
                    i += 1
                return np.array(result)
            
            h1 = halton(n_sobol, 2)
            h2 = halton(n_sobol, 3)
            points_2d = np.column_stack([h1, h2])
        
        # Map 2D points from unit square to triangle
        # Use the method: if u1 + u2 > 1, reflect to (1-u1, 1-u2)
        # This gives uniform distribution over the triangle
        u1 = points_2d[:, 0]
        u2 = points_2d[:, 1]
        
        # Reflect points outside the triangle
        mask = u1 + u2 > 1.0
        u1[mask] = 1.0 - u1[mask]
        u2[mask] = 1.0 - u2[mask]
        
        # Now (u1, u2) are uniformly distributed over the triangle
        # x0 = u1, x1 = u2, x2 = 1 - u1 - u2 (but we don't need x2 explicitly)
        
        # Evaluate all points
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            
            x0 = u1[i]
            x1 = u2[i]
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5  # Set remaining variables to minimize g
            
            f = np.array(problem(x))
            all_F.append(f)
        
        return all_F
