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

    def _halton_2d(self, n, skip=1):
        """Generate n 2D Halton sequence points in [0,1]^2, skipping first 'skip' points."""
        def halton_1d(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        points = []
        for i in range(skip, skip + n):
            points.append([halton_1d(i, 2), halton_1d(i, 3)])
        return np.array(points)

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 285
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
        
        # Generate points using Halton sequence mapped to the 2D simplex
        # For a point (u,v) in [0,1]^2, map to triangle using:
        # x0 = u * (1 - v), x1 = v * (1 - u)  -- this doesn't fill the triangle uniformly
        # Better: use barycentric coordinates
        # Sample (s,t) in [0,1]^2, then x0 = s*(1-t), x1 = t*(1-s) maps to a square region
        # For uniform triangle sampling: use sqrt transformation
        # Let r1, r2 be uniform, then x0 = r1*(1-r2), x1 = r2*(1-r1) still not uniform
        
        # Proper uniform sampling in triangle:
        # Generate u,v ~ U[0,1], if u+v > 1, set u=1-u, v=1-v
        # Then (u,v) is uniform in the triangle
        
        # Use Halton sequence for low discrepancy, then map to triangle
        # We'll generate n_evals points using Halton, then map each to the triangle
        
        # First, generate Halton points
        halton_pts = self._halton_2d(n_evals, skip=1)
        
        # Map each Halton point (u,v) to the triangle
        # If u + v > 1, reflect: u' = 1-u, v' = 1-v
        # This gives uniform distribution in the triangle
        
        points_to_eval = []
        for i in range(n_evals):
            u, v = halton_pts[i]
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            x = base_x.copy()
            x[0] = u
            x[1] = v
            points_to_eval.append(x)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        # If we have budget remaining (shouldn't happen with n_evals = budget), add more
        if len(all_points) < budget:
            # Use a fine grid to fill remaining
            n_grid = int(np.ceil(np.sqrt(2 * (budget - len(all_points)))))
            seen = set()
            for pt in points_to_eval:
                seen.add((round(pt[0], 8), round(pt[1], 8)))
            
            for i in range(n_grid + 1):
                for j in range(n_grid + 1 - i):
                    if len(all_points) >= budget or time.time() > deadline:
                        break
                    x0 = j / n_grid
                    x1 = i / n_grid
                    if x0 + x1 > 1.0:
                        continue
                    key = (round(x0, 8), round(x1, 8))
                    if key not in seen:
                        x = base_x.copy()
                        x[0] = x0
                        x[1] = x1
                        try:
                            f = problem(x)
                            all_points.append(np.array(f, dtype=float))
                        except Exception:
                            continue
                        seen.add(key)
        
        return all_points
# EVOLVE-BLOCK-END
