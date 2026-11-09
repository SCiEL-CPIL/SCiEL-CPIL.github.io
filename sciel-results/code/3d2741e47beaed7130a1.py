import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 specific structure:
        # The last (n_obj - 1) variables control the position on the Pareto front.
        # The first (dim - (n_obj - 1)) variables control the distance to the front.
        # For WFG9, setting the distance variables to the midpoint of their bounds
        # typically places the solution on the ideal Pareto front.
        
        n_pos = self.n_obj - 1  # 2 position variables
        n_dist = self.dim - n_pos # 10 distance variables
        
        # Set distance variables to midpoints
        mid = (lo + hi) / 2.0
        base_x = np.empty(self.dim)
        base_x[:n_dist] = mid[:n_dist]
        
        # We will sweep the last n_pos variables (indices n_dist to dim-1)
        # Determine grid size based on budget
        # Let N be the number of points per dimension in the position space.
        # Total evaluations = N^n_pos
        # We want N^n_pos <= budget
        # n_pos = 2, so N^2 <= 200 => N <= 14.14 => N=14
        
        N = int(np.floor(self.budget ** (1.0 / n_pos)))
        # Ensure we don't exceed budget
        if N < 2:
            N = 2
            
        # Create grid points for position variables
        # Use linspace to cover the range well
        pos_ranges = []
        for i in range(n_dist, self.dim):
            pos_ranges.append(np.linspace(lo[i], hi[i], N))
        
        # Generate all combinations
        # If n_pos is small (2), we can use meshgrid
        if n_pos == 2:
            g1, g2 = np.meshgrid(pos_ranges[0], pos_ranges[1], indexing='ij')
            pos_points = np.stack([g1.flatten(), g2.flatten()], axis=1)
        else:
            # General case for more position variables (not expected for n_obj=3)
            grids = np.meshgrid(*pos_ranges, indexing='ij')
            pos_points = np.stack([g.flatten() for g in grids], axis=1)
        
        # Check if we have exceeded budget
        num_points = pos_points.shape[0]
        if num_points > self.budget:
            # Reduce N or subsample
            # Simple reduction: take first budget points
            pos_points = pos_points[:self.budget]
            num_points = self.budget

        # Evaluate all points
        F_list = []
        X_list = []
        
        for i in range(num_points):
            if time.time() > deadline:
                break
                
            x = base_x.copy()
            x[n_dist:] = pos_points[i]
            
            try:
                f = problem(x)
                f = np.array(f)
                # Check if valid (finite)
                if np.all(np.isfinite(f)):
                    F_list.append(f)
                    X_list.append(x)
            except Exception:
                continue
                
        # If we have budget left, we could add random perturbations around the best points
        # or just return what we have. The grid is usually sufficient for WFG9.
        # However, to be robust, if we have significant budget left, we might do a local search.
        # But the prompt says "Deterministic grid... outperforming random/evolutionary".
        # So we stick to the grid.
        
        if not F_list:
            # Fallback: random sampling if grid failed
            rng = np.random.default_rng(0)
            for _ in range(self.budget):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                try:
                    f = problem(x)
                    f = np.array(f)
                    if np.all(np.isfinite(f)):
                        F_list.append(f)
                        X_list.append(x)
                except Exception:
                    continue

        return X_list
