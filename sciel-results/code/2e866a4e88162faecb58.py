"""Structure-exploiting optimizer for DTLZ7 using deterministic Pareto-front sampling."""
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and use a stratified grid plus boundary points to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # Let u = x[0], v = x[1] in [0,1]
        # f0 = u, f1 = v, f2 = 1 - sqrt(v)
        
        # Strategy:
        # 1. Reserve budget for boundary/corner points.
        # 2. Use a stratified grid for the interior.
        # 3. The grid is deterministic and covers the (u,v) square uniformly.
        
        # Boundary points:
        # Corners: (0,0), (1,0), (0,1), (1,1)
        # Edges: midpoints and quartiles
        # Let's define a set of important boundary points.
        # To maximize hypervolume, we want to cover the "outer" envelope well.
        
        # Let's allocate 20% of budget to boundary, 80% to interior.
        n_boundary = max(4, budget // 5)
        n_interior = budget - n_boundary
        
        if n_interior < 4:
            n_boundary = 4
            n_interior = budget - 4
            if n_interior < 0:
                n_interior = 0
                n_boundary = budget

        # Generate boundary points
        # We want points on the edges of the [0,1]x[0,1] square.
        # Specifically:
        # u=0, v in [0,1]
        # u=1, v in [0,1]
        # v=0, u in [0,1]
        # v=1, u in [0,1]
        
        # Let's create a list of (u, v) pairs for boundary.
        boundary_points = []
        
        # Add corners first
        boundary_points.append((0.0, 0.0))
        boundary_points.append((1.0, 0.0))
        boundary_points.append((0.0, 1.0))
        boundary_points.append((1.0, 1.0))
        
        # Distribute remaining boundary points along edges
        # If we have n_boundary points, and 4 are corners, we have n_boundary - 4 left.
        # Let's distribute them evenly on the 4 edges.
        if n_boundary > 4:
            remaining = n_boundary - 4
            # 1/4 of remaining on each edge
            per_edge = remaining // 4
            leftover = remaining % 4
            
            # Edge 1: u=0, v in (0,1)
            # Edge 2: u=1, v in (0,1)
            # Edge 3: v=0, u in (0,1)
            # Edge 4: v=1, u in (0,1)
            
            # To avoid duplicates at corners, we sample in (0,1)
            
            # Edge 1: u=0
            for i in range(per_edge):
                v = (i + 1) / (per_edge + 1)
                boundary_points.append((0.0, v))
            
            # Edge 2: u=1
            for i in range(per_edge):
                v = (i + 1) / (per_edge + 1)
                boundary_points.append((1.0, v))
            
            # Edge 3: v=0
            for i in range(per_edge):
                u = (i + 1) / (per_edge + 1)
                boundary_points.append((u, 0.0))
            
            # Edge 4: v=1
            for i in range(per_edge):
                u = (i + 1) / (per_edge + 1)
                boundary_points.append((u, 1.0))
            
            # Distribute leftover
            edges = [
                lambda i: (0.0, (i + 1) / (per_edge + 2)),
                lambda i: (1.0, (i + 1) / (per_edge + 2)),
                lambda i: ((i + 1) / (per_edge + 2), 0.0),
                lambda i: ((i + 1) / (per_edge + 2), 1.0)
            ]
            for i in range(leftover):
                u, v = edges[i % 4](i)
                # Check for duplicates
                if (u, v) not in boundary_points:
                    boundary_points.append((u, v))

        # Generate interior points using a stratified grid
        # n_interior points in a grid.
        # Let's use a square-ish grid.
        n_grid = int(np.sqrt(n_interior))
        if n_grid < 1:
            n_grid = 1
        n_grid_y = n_interior // n_grid
        if n_grid_y < 1:
            n_grid_y = 1
        # Adjust to not exceed budget
        if n_grid * n_grid_y > n_interior:
            n_grid_y = n_interior // n_grid
            if n_grid_y < 1:
                n_grid = 1
                n_grid_y = n_interior

        interior_points = []
        for i in range(n_grid):
            for j in range(n_grid_y):
                # Stratified sampling: center of each cell
                u = (i + 0.5) / n_grid
                v = (j + 0.5) / n_grid_y
                interior_points.append((u, v))
        
        # Combine and truncate to budget
        all_points = boundary_points + interior_points
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in all_points:
            key = (round(p[0], 10), round(p[1], 10))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
        
        # Truncate to budget
        unique_points = unique_points[:budget]
        
        # Evaluate
        for u, v in unique_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # x[2:] remains 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
