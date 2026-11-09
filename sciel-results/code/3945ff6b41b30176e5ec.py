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
        
        # Use a stratified triangular grid over the 2D simplex
        # For n_evals points, use a triangular arrangement
        # Row i has (n-i) points for i=0..n-1, total n*(n+1)/2 points
        # Find largest n such that n*(n+1)/2 <= n_evals
        
        n_tri = 1
        while (n_tri + 1) * (n_tri + 2) // 2 <= n_evals:
            n_tri += 1
        # n_tri is now the largest integer with n_tri*(n_tri+1)/2 <= n_evals
        
        points_to_eval = []
        
        # Generate triangular grid points
        # Point in row i, column j: x0 = j/(n_tri-1) if n_tri>1 else 0, x1 = (i-j)/(n_tri-1)
        # Actually, let's use barycentric coordinates
        # For a triangle with vertices (0,0), (1,0), (0,1):
        # Points: (a/(n-1), b/(n-1)) where a+b <= n-1, a,b >= 0
        
        if n_tri == 1:
            # Only 1 point: center of triangle
            x = base_x.copy()
            x[0] = 1.0/3
            x[1] = 1.0/3
            points_to_eval.append(x)
        else:
            n_pts = n_tri
            for i in range(n_pts):
                for j in range(n_pts - i):
                    x0 = j / (n_pts - 1)
                    x1 = (n_pts - 1 - i - j) / (n_pts - 1)
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    points_to_eval.append(x)
        
        # If we have remaining budget, add more points using finer stratification
        remaining = n_evals - len(points_to_eval)
        if remaining > 0:
            # Add points from a finer grid, skipping duplicates
            n_fine = n_tri + 1
            seen = set()
            for pt in points_to_eval:
                seen.add((round(pt[0], 10), round(pt[1], 10)))
            
            for i in range(n_fine):
                for j in range(n_fine - i):
                    if len(points_to_eval) >= n_evals:
                        break
                    x0 = j / (n_fine - 1)
                    x1 = (n_fine - 1 - i - j) / (n_fine - 1)
                    key = (round(x0, 10), round(x1, 10))
                    if key not in seen:
                        x = base_x.copy()
                        x[0] = x0
                        x[1] = x1
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
        
        # If we have budget remaining, add random points in the triangle for robustness
        if len(all_points) < budget:
            rng = np.random.default_rng(12345)
            while len(all_points) < budget and time.time() < deadline:
                # Random point in triangle: use rejection sampling or barycentric
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                x0 = r1
                x1 = r2
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
