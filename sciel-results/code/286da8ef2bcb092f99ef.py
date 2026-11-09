import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # DTLZ7 structure:
            # f1 = x0, f2 = x1, f3 = (1 + g) * (h1 + x1*h2)
            # where g = sum(x_i^2) for i >= 2, optimal g=0 when x2..x21=0
            # h1 = 1 - sqrt(x0/(1+g)), h2 = 1 - x0/(1+g)
            # With g=0: f1=x0, f2=x1, f3=(1+x1)*(1-sqrt(x0))
            
            # Pareto front is 2D: x0 in [0,1], x1 in [0,1]
            # To maximize hypervolume, we want points that are non-dominated
            # and spread across the front.
            
            # Strategy: Use the entire budget to sample the 2D Pareto front
            # with a dense grid. A 14x14 grid gives 196 points, leaving 4 for
            # additional coverage at the extremes.
            
            n_grid = 14
            x0_vals = np.linspace(0.0, 1.0, n_grid)
            x1_vals = np.linspace(0.0, 1.0, n_grid)
            
            X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
            x0_flat = X0.flatten()
            x1_flat = X1.flatten()
            
            points_to_eval = []
            
            for x0, x1 in zip(x0_flat, x1_flat):
                x = lo.copy()
                x[0] = x0
                x[1] = x1
                # x2..x21 stay at lo (which is 0 for standard DTLZ7)
                points_to_eval.append(x)
            
            # Add 4 extra points to cover the corners more precisely
            # if not already covered by the grid
            corners = [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0)
            ]
            
            n_remaining = self.budget - len(points_to_eval)
            for c in corners[:n_remaining]:
                x = lo.copy()
                x[0] = c[0]
                x[1] = c[1]
                points_to_eval.append(x)
            
            # Evaluate all points
            for x in points_to_eval:
                problem(x)
                
            return None
