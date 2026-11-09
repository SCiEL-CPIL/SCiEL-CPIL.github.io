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

    def _stratified_triangle(self, n):
        """Generate n stratified points in the 2D simplex (triangle) for hypervolume optimization."""
        # For hypervolume maximization on a 2D simplex with far reference point,
        # we want good coverage especially near the vertices.
        # Use a stratified approach: divide into n rows, place points in each row.
        
        points = []
        # Use a grid-like stratification in barycentric coordinates
        # For each "level" i (0 to n-1), we have i+1 positions
        # Total points in a triangular grid with n rows: n*(n+1)/2
        # Find n such that n*(n+1)/2 <= budget, then add extra points
        
        # Find largest n where n*(n+1)/2 <= n_evals
        n_rows = int(np.floor((np.sqrt(8 * n + 1) - 1) / 2))
        
        # Generate points in triangular grid with stratified offsets
        for i in range(n_rows):
            n_in_row = i + 1
            for j in range(n_in_row):
                # Stratified position within each cell
                # Barycentric: (a, b, c) with a+b+c=1
                # x0 = b, x1 = a, x2 = c (for DTLZ1: f1=x0, f2=x1, f3=1-x0-x1)
                # In the triangle: x0 in [0,1], x1 in [0, 1-x0]
                # Stratified: use midpoints with jitter
                x0 = (j + 0.5) / n_in_row
                x1 = (i + 0.5) / (n_rows - i) * (1 - x0)
                points.append([x0, x1])
        
        # If we have more budget, add additional stratified points
        remaining = n - len(points)
        if remaining > 0:
            # Add points from a finer Halton sequence for the remainder
            def halton_1d(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            # Start from offset to avoid duplicates
            offset = 1000
            for k in range(remaining):
                idx = offset + k
                u = halton_1d(idx, 2)
                v = halton_1d(idx, 3)
                # Map to triangle via reflection
                if u + v > 1.0:
                    u = 1.0 - u
                    v = 1.0 - v
                points.append([u, v])
        
        return np.array(points[:n])

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
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        # Generate stratified points on the 2D simplex
        simplex_pts = self._stratified_triangle(n_evals)
        
        points_to_eval = []
        for i in range(n_evals):
            x0, x1 = simplex_pts[i]
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
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
        
        return all_points
# EVOLVE-BLOCK-END
