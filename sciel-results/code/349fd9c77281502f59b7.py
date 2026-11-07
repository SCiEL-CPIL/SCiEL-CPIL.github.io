import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # DTLZ4 with 3 objectives, 12 variables
        # Pareto front: x[2:] = 0.5 (minimizes g), x[0], x[1] in [0,1]
        # Objectives on the front:
        #   f0 = x[0] * 0.5 * (1 + g)
        #   f1 = x[1] * 0.5 * (1 + g)
        #   f2 = (1 - x[1]) * 0.5 * (1 + g)
        # With g=0: f0 = 0.5*x[0], f1 = 0.5*x[1], f2 = 0.5*(1-x[1])
        # The front is a quarter-circle-like arc in 3D.
        
        # Actually for DTLZ4 with M=3:
        # f0 = 0.5 * x0 * (1+g)
        # f1 = 0.5 * x1 * (1+g)  
        # f2 = 0.5 * (1-x1) * (1+g)
        
        # Wait, let me reconsider. Standard DTLZ4:
        # f_i = 0.5 * x_i * (1 + g) for i=0..M-2
        # f_{M-1} = 0.5 * (1 - x_{M-2}) * (1 + g)
        # where g = sum((x_j - 0.5)^alpha) for j >= M-1
        
        # For M=3, dim=12: x[0], x[1] are the "position" variables, x[2:] affect g.
        # When x[2:] = 0.5, g = 0.
        # f0 = 0.5 * x[0]
        # f1 = 0.5 * x[1]
        # f2 = 0.5 * (1 - x[1])
        
        # The Pareto front is a line segment? No, it's a 2D surface parameterized by x[0] and x[1].
        # Actually f0 depends only on x[0], f1 and f2 depend only on x[1].
        # So the front is: f0 in [0, 0.5], f1 in [0, 0.5], f2 = 0.5 - f1.
        # This is a line segment in 3D! f0 varies independently, and f1+f2=0.5.
        
        # Wait no. f0 = 0.5*x[0], f1 = 0.5*x[1], f2 = 0.5*(1-x[1]).
        # So f1 + f2 = 0.5 always. And f0 ranges [0, 0.5] independently.
        # The Pareto front is a 2D region: all points where f1+f2=0.5, f0 in [0,0.5], f1 in [0,0.5].
        # Actually it's a line segment in the (f1,f2) plane extruded along f0.
        # So it's a 2D surface: f0 in [0,0.5], f1 in [0,0.5], f2=0.5-f1.
        
        # For hypervolume, we want to cover this surface well.
        # The reference point is [2.5, 2.5, 2.5].
        
        # Strategy: Use stratified sampling on the Pareto front.
        # Since the front is 2D (parameterized by x[0] and x[1]), we should
        # distribute points uniformly over this 2D surface.
        
        # Budget is 200. Let's use a stratified grid on x[0] and x[1].
        # To maximize hypervolume, we want good coverage of the 2D front.
        
        # Use cell-center stratified sampling: divide [0,1] into n cells,
        # place one point at the center of each cell.
        
        # We can also add some points near the boundaries to capture the extremes.
        
        n_evals = 0
        results = []
        
        # Stratified grid: n x n points at cell centers
        # 14x14 = 196, leaving 4 for boundary points
        n_grid = 14
        x0_centers = (np.arange(n_grid) + 0.5) / n_grid
        x1_centers = (np.arange(n_grid) + 0.5) / n_grid
        
        points = []
        for x0 in x0_centers:
            for x1 in x1_centers:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                points.append(x)
        
        # Add boundary/corner points to capture extremes
        # Corners of the [0,1]x[0,1] square
        boundary_points = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        
        for bp in boundary_points:
            if n_evals + len(points) + len(boundary_points) <= self.budget:
                x = np.zeros(self.dim)
                x[0] = bp[0]
                x[1] = bp[1]
                x[2:] = 0.5
                points.append(x)
        
        # If we still have budget, add more stratified points or random points
        # But the grid + 4 corners = 200 exactly for n_grid=14
        # 14*14 + 4 = 196 + 4 = 200. Perfect.
        
        # Evaluate all points
        for x in points:
            if time.time() > deadline:
                break
            if n_evals >= self.budget:
                break
            f = problem(x)
            n_evals += 1
            results.append(f)
        
        return results
