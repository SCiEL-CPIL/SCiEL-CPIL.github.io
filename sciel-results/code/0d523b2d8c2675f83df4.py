import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Deterministic Grid Search (3 levels per dimension: -5, 0, 5)
        # This creates a 3^dim grid. For dim=5, that's 243 points.
        # This provides a much better initial estimate than just vertices.
        levels = [-5.0, 0.0, 5.0]
        grids = np.meshgrid(*([levels] * self.dim), indexing='ij')
        points = np.stack(grids, axis=-1).reshape(-1, self.dim)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        for p in points:
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        x = best_x
        fx = best_f
        
        # 2. Local Hill Climb (Best-Step with Decaying Step)
        remaining = self.budget - calls
        if remaining <= 0:
            return x, fx
            
        # The grid spacing is 5.0. A good starting step for refinement is half the spacing.
        step = 2.5 
        min_step = 1e-6
        shrink_factor = 0.5
        
        while remaining > 0 and step > min_step:
            best_neighbor_x = None
            best_neighbor_val = fx
            
            # Try all 2*dim directions
            for i in range(self.dim):
                # Positive direction
                if remaining > 0:
                    y = x.copy()
                    y[i] += step
                    fy = func(y)
                    remaining -= 1
                    if fy < best_neighbor_val:
                        best_neighbor_val = fy
                        best_neighbor_x = y.copy()
                        
                # Negative direction
                if remaining > 0:
                    y = x.copy()
                    y[i] -= step
                    fy = func(y)
                    remaining -= 1
                    if fy < best_neighbor_val:
                        best_neighbor_val = fy
                        best_neighbor_x = y.copy()
                        
            if best_neighbor_x is not None:
                x = best_neighbor_x
                fx = best_neighbor_val
            else:
                step *= shrink_factor

        return x, fx
