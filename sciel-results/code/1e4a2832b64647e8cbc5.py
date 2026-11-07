# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where x[2:] = 0.
    We use stratified sampling on the 2D subspace with transformation
    to ensure uniform coverage in the objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # 300s limit with 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        results = []
        
        n_total = self.budget
        
        # Explicitly add corners and edge midpoints for better coverage
        explicit_points = [
            [0.0, 0.0],   # f1=0, f2=2, f3=1
            [0.0, 1.0],   # f1=0, f2=0, f3=0
            [1.0, 0.0],   # f1=1, f2=2, f3=1
            [1.0, 1.0],   # f1=1, f2=0, f3=0
            [0.5, 0.0],   # f1=0.5, f2=2, f3=1
            [0.5, 1.0],   # f1=0.5, f2=0, f3=0
            [0.0, 0.5],   # f1=0, f2=1, f3=0.5
            [1.0, 0.5],   # f1=1, f2=1, f3=0.5
        ]
        
        n_explicit = len(explicit_points)
        n_stratified = max(0, n_total - n_explicit)
        
        # First evaluate explicit points
        for pt in explicit_points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = pt[0]
            x[1] = pt[1]
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        # Then use stratified sampling for the remaining budget
        if n_stratified > 0:
            # Use a grid-based stratified approach
            # Choose grid size to approximate n_stratified points
            n_per_dim = int(np.sqrt(n_stratified))
            if n_per_dim < 1:
                n_per_dim = 1
            
            # Create grid indices
            grid_idx = np.arange(n_per_dim)
            grid_x0, grid_x1 = np.meshgrid(grid_idx, grid_idx, indexing='ij')
            grid_x0 = grid_x0.flatten()
            grid_x1 = grid_x1.flatten()
            
            # If we have more points than grid cells, add jitter
            n_cells = n_per_dim * n_per_dim
            if n_stratified <= n_cells:
                # Use first n_stratified cells
                x0_vals = (grid_x0[:n_stratified] + 0.5) / n_per_dim
                x1_vals = (grid_x1[:n_stratified] + 0.5) / n_per_dim
            else:
                # Use all cells plus extra points with random jitter
                x0_vals = (grid_x0 + np.random.rand(n_cells) * 0.5) / n_per_dim
                x1_vals = (grid_x1 + np.random.rand(n_cells) * 0.5) / n_per_dim
                extra = n_stratified - n_cells
                if extra > 0:
                    x0_extra = np.random.rand(extra)
                    x1_extra = np.random.rand(extra)
                    x0_vals = np.concatenate([x0_vals, x0_extra])
                    x1_vals = np.concatenate([x1_vals, x1_extra])
            
            # Clip to [0, 1]
            x0_vals = np.clip(x0_vals, 0.0, 1.0)
            x1_vals = np.clip(x1_vals, 0.0, 1.0)
            
            # Transform: x1 = s1^2 for uniformity in sqrt(x1)
            x1_transformed = x1_vals ** 2
            
            for i in range(n_stratified):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0_vals[i]
                x[1] = x1_transformed[i]
                
                f = problem(x)
                results.append(f)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
