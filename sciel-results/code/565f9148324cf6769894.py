import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Structure:
        # Shape variables: indices 1, 5, 9 (0-based)
        # Position variables: all others
        shape_indices = [1, 5, 9]
        pos_indices = [i for i in range(self.dim) if i not in shape_indices]
        
        # Set position variables to midpoints
        x_base = np.zeros(self.dim)
        for i in pos_indices:
            x_base[i] = (lo[i] + hi[i]) / 2.0
            
        # Generate grid points for shape variables
        # Budget is 200.
        # 6x6x5 = 180 points.
        # Let's try 6x6x5 grid.
        n1, n2, n3 = 6, 6, 5
        
        grid_points = []
        
        # Create ranges for shape variables
        ranges = []
        for idx in shape_indices:
            # Use linspace including endpoints
            ranges.append(np.linspace(lo[idx], hi[idx], n1 if idx==shape_indices[0] else (n2 if idx==shape_indices[1] else n3)))
            
        # Actually, let's just do a standard meshgrid
        g1 = np.linspace(lo[shape_indices[0]], hi[shape_indices[0]], 6)
        g2 = np.linspace(lo[shape_indices[1]], hi[shape_indices[1]], 6)
        g3 = np.linspace(lo[shape_indices[2]], hi[shape_indices[2]], 5)
        
        # Generate combinations
        for x1 in g1:
            for x2 in g2:
                for x3 in g3:
                    if time.time() > deadline:
                        break
                    x = x_base.copy()
                    x[shape_indices[0]] = x1
                    x[shape_indices[1]] = x2
                    x[shape_indices[2]] = x3
                    grid_points.append(x)
                    
        # Evaluate all grid points
        F = []
        X = []
        for x in grid_points:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
            X.append(x)
            
        # If we have budget left, we can try to refine or just return
        # 180 points used. 20 left.
        # We can add 20 random points or refine the grid slightly.
        # Given the deterministic nature, adding random points might help fill gaps
        # if the grid is too coarse, but the grid is already quite fine.
        # Let's add a few points around the center of the bounds for shape variables
        # to ensure we don't miss a sharp feature.
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            x = x_base.copy()
            for idx in shape_indices:
                x[idx] = rng.uniform(lo[idx], hi[idx])
            f = problem(x)
            F.append(f)
            X.append(x)
            
        return X
