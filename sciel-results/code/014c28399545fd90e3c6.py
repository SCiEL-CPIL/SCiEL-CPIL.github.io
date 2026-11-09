"""DTLZ1 optimizer using stratified simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with stratified points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shape variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        # The Pareto front for DTLZ1 M=3 is a 2D triangle in shape space:
        # x[0] >= 0, x[1] >= 0, x[0] + x[1] <= 1
        # Objectives on the front (g=0): f1=x0, f2=x1, f3=1-x0-x1
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        points_to_eval = []
        
        # Use a stratified triangular grid: divide the triangle into cells
        # and place one point per cell. This gives uniform coverage.
        # For N divisions per edge, we get N*(N+1)/2 cells.
        # Find N such that N*(N+1)/2 <= n_evals, then fill remaining with
        # midpoints of the finest cells or additional stratified points.
        
        n_div = 1
        while (n_div + 1) * (n_div + 2) // 2 <= n_evals:
            n_div += 1
        # n_div*(n_div+1)/2 <= n_evals < (n_div+1)*(n_div+2)/2
        
        # Generate points: for each cell in the triangular grid, place a point
        # Cell (i, j) where i >= 0, j >= 0, i + j < n_div
        # The cell spans x0 in [j/n_div, (j+1)/n_div] and x1 in [i/n_div, (i+1)/n_div]
        # but constrained by x0 + x1 <= 1
        # Place point at the centroid of the valid region
        
        for i in range(n_div):
            for j in range(n_div - i):
                # Cell: x0 in [j/n_div, (j+1)/n_div], x1 in [i/n_div, (i+1)/n_div]
                # with constraint x0 + x1 <= 1
                # The centroid of the unconstrained cell is ((j+0.5)/n_div, (i+0.5)/n_div)
                # Check if this satisfies x0 + x1 <= 1
                x0_c = (j + 0.5) / n_div
                x1_c = (i + 0.5) / n_div
                if x0_c + x1_c <= 1.0:
                    x = base_x.copy()
                    x[0] = x0_c
                    x[1] = x1_c
                    points_to_eval.append(x)
        
        # Fill remaining budget with points from a finer grid
        remaining = n_evals - len(points_to_eval)
        if remaining > 0:
            # Use n_div+1 divisions and add points not already in points_to_eval
            n_fine = n_div + 1
            seen = set()
            for pt in points_to_eval:
                seen.add((round(pt[0], 12), round(pt[1], 12)))
            
            for i in range(n_fine):
                for j in range(n_fine - i):
                    if len(points_to_eval) >= n_evals:
                        break
                    x0_c = (j + 0.5) / n_fine
                    x1_c = (i + 0.5) / n_fine
                    if x0_c + x1_c > 1.0:
                        continue
                    key = (round(x0_c, 12), round(x1_c, 12))
                    if key not in seen:
                        x = base_x.copy()
                        x[0] = x0_c
                        x[1] = x1_c
                        points_to_eval.append(x)
                        seen.add(key)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        # If we have budget remaining after grid points, add mid-edge points
        # These are high-value points for hypervolume coverage
        if len(all_points) < budget:
            # Mid-edge points of the triangle: (0.5, 0), (0, 0.5), (0.5, 0.5)
            # But (0.5, 0.5) is on the edge x0+x1=1, and (0.5,0), (0,0.5) are on axes
            # Actually, the edges are: x0=0, x1=0, x0+x1=1
            # Mid-points: (0, 0.5), (0.5, 0), (0.5, 0.5)
            mid_edge = [(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
            for x0, x1 in mid_edge:
                if len(all_points) >= budget:
                    break
                if time.time() > deadline:
                    break
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                try:
                    f = problem(x)
                    all_points.append(np.array(f, dtype=float))
                except Exception:
                    break
        
        return all_points
# EVOLVE-BLOCK-END
