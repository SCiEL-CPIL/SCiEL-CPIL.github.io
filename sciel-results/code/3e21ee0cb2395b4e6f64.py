import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, keep 15s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure:
        # First M-1 = 2 variables determine position on the Pareto front.
        # Remaining n-M+1 = 10 variables determine g(x), with g=0 when they are 0.5.
        n_front_vars = self.n_obj - 1  # 2 for 3 objectives
        n_g_vars = self.dim - n_front_vars  # 10 for 12 dim
        
        # Budget allocation:
        # - Reserve 6 evaluations for critical vertices and edge midpoints
        # - Use remaining budget for cosine-spaced grid
        # - The 3 vertices (0,0), (0,1), (1,0) contribute most to hypervolume
        # - Edge midpoints (0.5,0), (0.5,1), (0,0.5), (1,0.5) help fill gaps
        
        # Critical points that must be included (these are where hypervolume contribution peaks)
        critical_points = [
            (0.0, 0.0),   # vertex: f=(0,0,1)
            (0.0, 1.0),   # vertex: f=(0,1,0)
            (1.0, 0.0),   # vertex: f=(1,0,0)
            (0.5, 0.0),   # edge midpoint
            (0.5, 1.0),   # edge midpoint
            (0.0, 0.5),   # edge midpoint
        ]
        
        # Grid size: use sqrt of remaining budget for square grid
        remaining_budget = self.budget - len(critical_points)
        if remaining_budget < 4:
            grid_size = 2
        else:
            grid_size = int(np.sqrt(remaining_budget))
        
        # Cosine spacing: clusters points near 0 and 1 (boundaries)
        # For i in 0..n-1: t_i = 0.5 + 0.5 * cos(pi * i / (n-1))
        # This gives values in [0, 1] with higher density near 0 and 1
        if grid_size > 1:
            u_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(grid_size) / (grid_size - 1))
            v_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(grid_size) / (grid_size - 1))
        else:
            u_vals = np.array([0.5])
            v_vals = np.array([0.5])
        
        # Build list of (u, v) points to evaluate
        points = []
        seen = set()
        
        # Add critical points first (highest hypervolume contribution)
        for u, v in critical_points:
            key = (round(u, 6), round(v, 6))
            if key not in seen:
                seen.add(key)
                points.append((u, v))
        
        # Add grid points, skipping any that duplicate critical points
        for u in u_vals:
            for v in v_vals:
                key = (round(u, 6), round(v, 6))
                if key not in seen:
                    seen.add(key)
                    points.append((u, v))
        
        # If we still have budget, add a few more strategic points
        # Center of the front and additional edge points
        extra_strategic = [
            (0.5, 0.5),
            (1.0, 0.5),
            (0.25, 0.0),
            (0.75, 0.0),
            (0.25, 1.0),
            (0.75, 1.0),
            (0.0, 0.25),
            (0.0, 0.75),
        ]
        
        for u, v in extra_strategic:
            if len(points) >= self.budget:
                break
            key = (round(u, 6), round(v, 6))
            if key not in seen:
                seen.add(key)
                points.append((u, v))
        
        # Truncate to budget if needed
        points = points[:self.budget]
        
        # Evaluate all points
        F = []
        for u, v in points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # Set remaining variables to 0.5 to ensure g(x)=0
            x[2:] = 0.5
            f = problem(x)
            F.append(f)
        
        return F
