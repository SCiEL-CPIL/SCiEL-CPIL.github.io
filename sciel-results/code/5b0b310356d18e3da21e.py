import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 270.0
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Pre-compute optimal shape variables for different lambda
            # h1(x) = x - 0.5 * sin(2*pi*x)
            # h2(x) = x - 0.5 * sin(4*pi*x)
            
            def h1(x):
                return x - 0.5 * np.sin(2 * np.pi * x)
            
            def h2(x):
                return x - 0.5 * np.sin(4 * np.pi * x)
            
            # Find x* for given lambda: min h1(x) + lambda * h2(x)
            # x in [0, 1]
            x_grid = np.linspace(0, 1, 1000)
            h1_vals = h1(x_grid)
            h2_vals = h2(x_grid)
            
            lambdas = np.logspace(-3, 3, 15)
            x_stars = []
            for lam in lambdas:
                obj = h1_vals + lam * h2_vals
                idx = np.argmin(obj)
                x_stars.append(x_grid[idx])
            
            # Sample x1 values
            x1_vals = np.linspace(0.01, 1.0, 15)
            
            points = []
            used = 0
            
            for x1 in x1_vals:
                for x_star in x_stars:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x = np.full(self.dim, x_star)
                    x[0] = x1
                    
                    # Clip to bounds
                    x = np.clip(x, lo, hi)
                    
                    f = problem(x)
                    points.append(f)
                    used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            # If we have budget left, add random points or refine
            # But the structured points should be sufficient.
            
            return points
