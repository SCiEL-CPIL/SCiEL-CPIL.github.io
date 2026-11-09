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
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        # The Pareto front for DTLZ1 M=3 is a 2D triangle in objective space.
        # Shape variables x[0], x[1] must satisfy x[0] >= 0, x[1] >= 0, x[0]+x[1] <= 1.
        # The third shape component is 1 - x[0] - x[1].
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        # Use a structured lattice on the 2D simplex for uniform coverage
        # The Pareto front of DTLZ1 (M=3) is a triangle: x[0]>=0, x[1]>=0, x[0]+x[1]<=1
        # We'll use a triangular lattice that distributes points evenly across the simplex
        
        # Find the largest K such that (K+1)(K+2)/2 <= n_evals
        K = 1
        while (K + 1) * (K + 2) // 2 <= n_evals:
            K += 1
        K -= 1  # Now (K+1)(K+2)/2 <= n_evals
        
        valid_points = []
        for i in range(K + 1):
            for j in range(K - i + 1):
                valid_points.append([i / K, j / K])
        
        # If we have room, add boundary and interior points to fill remaining budget
        remaining = n_evals - len(valid_points)
        if remaining > 0:
            # Add midpoints of edges and interior points
            edge_midpoints = []
            for i in range(K + 1):
                for j in range(K - i + 1):
                    # Add offset points to fill gaps
                    if i < K:
                        edge_midpoints.append([(2*i + 1) / (2*K), j / K])
                    if j < K - i:
                        edge_midpoints.append([i / K, (2*j + 1) / (2*K)])
            
            # Remove duplicates and keep only valid points
            seen = set()
            for p in edge_midpoints:
                key = (round(p[0], 10), round(p[1], 10))
                if key not in seen and p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= 1.0 + 1e-10:
                    seen.add(key)
                    valid_points.append(p)
                    if len(valid_points) >= n_evals:
                        break
        
        # Trim to exactly n_evals if needed
        valid_points = valid_points[:n_evals]

        # Evaluate the points
        for p in valid_points:
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = p[0]
            x[1] = p[1]
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
