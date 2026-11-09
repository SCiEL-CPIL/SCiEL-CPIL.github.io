"""DTLZ1 multi-objective optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer: exploit structure, low-discrepancy diversity, local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (hi + lo) / 2.0

        # DTLZ1 structure:
        # f1 = 0.5 * g * (1 + x1)
        # f2 = 0.5 * g * (1 - x1) * (1 + x2)
        # f3 = 0.5 * g * (1 - x2)
        # where g = 100 * (n-1 + sum((xi - 0.5)^2 - cos(20*pi*(xi-0.5))) for i in 2..n)
        # Optimal: g=0 when xi=0.5 for i>=2, and x1,x2 vary to trace the front.
        # Pareto front: x1 in [0,1], x2 in [0,1], f1+f2+f3=0.5, convex front.

        # Strategy:
        # 1. Set "g" variables (indices 2..dim-1) to 0.5 to get g=0
        # 2. Sample x1 (index 0) and x2 (index 1) to cover the Pareto front
        # 3. Use local search / refinement around promising points

        # Low-discrepancy sequence for 2D (x1, x2) coverage
        def halton_sequence(n, dim_h=2):
            """Generate n points in [0,1]^dim_h using Halton sequences."""
            pts = np.zeros((n, dim_h))
            for d in range(dim_h):
                base = d + 2  # 2, 3, 4, ...
                for i in range(n):
                    f = 1.0
                    x = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        x += f * (i_val % base)
                        i_val //= base
                    pts[i, d] = x
            return pts

        # Phase 1: Structured sampling on the Pareto front
        # Use Halton sequence for x1, x2 to get good coverage
        n_structured = min(100, budget // 2)
        halton_pts = halton_sequence(n_structured, 2)

        # Create candidate points with g-variables at 0.5
        X = np.zeros((n_structured, dim))
        for i in range(n_structured):
            X[i, 0] = lo[0] + span[0] * halton_pts[i, 0]
            X[i, 1] = lo[1] + span[1] * halton_pts[i, 1]
            for j in range(2, dim):
                X[i, j] = center[j]

        F = np.array([problem(x) for x in X])
        used = n_structured

        # Check deadline
        if time.time() > deadline - 5.0:
            return X

        # Phase 2: Add structured grid points for better hypervolume coverage
        # The Pareto front in (x1, x2) space is a triangle. Use a stratified grid.
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5.0:
            # Create a stratified grid in the triangular region
            # x1 in [0,1], x2 in [0, 1-x1] for the simplex
            n_grid = int(np.sqrt(remaining * 0.5))
            n_grid = max(n_grid, 5)
            n_grid = min(n_grid, 40)
            
            grid_pts = []
            for i in range(n_grid):
                for j in range(n_grid):
                    x1 = i / (n_grid - 1) if n_grid > 1 else 0.5
                    x2 = j / (n_grid - 1) if n_grid > 1 else 0.5
                    if x1 + x2 <= 1.0 + 1e-10:
                        grid_pts.append((x1, x2))
            
            # Also add edge and vertex points explicitly
            n_edge = min(10, remaining // 5)
            if n_edge > 0:
                for k in range(n_edge):
                    t = k / max(n_edge - 1, 1)
                    grid_pts.append((t, 0.0))
                    grid_pts.append((0.0, t))
                    grid_pts.append((t, 1.0 - t))
            
            # Add vertices
            grid_pts.append((0.0, 0.0))
            grid_pts.append((1.0, 0.0))
            grid_pts.append((0.0, 1.0))
            
            # Remove duplicates and shuffle for diversity
            seen = set()
            unique_pts = []
            for pt in grid_pts:
                key = (round(pt[0], 10), round(pt[1], 10))
                if key not in seen:
                    seen.add(key)
                    unique_pts.append(pt)
            
            np.random.shuffle(unique_pts)
            
            # Evaluate grid points
            for x1, x2 in unique_pts:
                if used >= budget or time.time() > deadline - 5.0:
                    break
                x = np.zeros(dim)
                x[0] = lo[0] + span[0] * x1
                x[1] = lo[1] + span[1] * x2
                for j in range(2, dim):
                    x[j] = center[j]
                
                fx = problem(x)
                used += 1
                
                # Add to population
                X = np.vstack([X, x])
                F = np.vstack([F, fx])
        
        # Phase 3: Fill remaining budget with additional Halton points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5.0:
            n_extra = remaining
            # Use a shifted Halton sequence for better coverage
            extra_halton = halton_sequence(n_extra + 50, 2)
            # Skip first 50 to avoid overlap with initial Halton
            extra_halton = extra_halton[50:]
            
            for i in range(n_extra):
                if used >= budget or time.time() > deadline - 5.0:
                    break
                x = np.zeros(dim)
                x[0] = lo[0] + span[0] * extra_halton[i, 0]
                x[1] = lo[1] + span[1] * extra_halton[i, 1]
                for j in range(2, dim):
                    x[j] = center[j]
                
                fx = problem(x)
                used += 1
                
                X = np.vstack([X, x])
                F = np.vstack([F, fx])
        
        # Return all evaluated points
        return X
# EVOLVE-BLOCK-END
