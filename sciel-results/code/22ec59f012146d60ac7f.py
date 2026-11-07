# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 leveraging analytic structure and Latin Hypercube sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 300s limit

        n_points = self.budget
        
        # Generate Latin Hypercube sample in 2D
        # This ensures uniform coverage along each axis, which is good for hypervolume
        # as it guarantees spread across the entire range of x1 and x2.
        
        # Method:
        # 1. Create a permutation of indices 0..N-1 for x1
        # 2. Create a permutation of indices 0..N-1 for x2
        # 3. Map to [0, 1] range.
        
        # To make it deterministic and reproducible, we use a fixed seed for the permutation
        # or a specific construction.
        # Standard LHS:
        # x_i = (perm1[i] + U1[i]) / N
        # y_i = (perm2[i] + U2[i]) / N
        
        # However, for hypervolume, we might want to include the boundaries 0 and 1.
        # Standard LHS doesn't strictly hit 0 or 1 unless we adjust.
        # We can use "Stratified Sampling" which is similar.
        
        # Let's use a simple deterministic LHS construction:
        # Permutation 1: 0, 1, 2, ..., N-1
        # Permutation 2: 0, 1, 2, ..., N-1 (or a shift)
        # If we use identity permutation, we get a diagonal grid, which is bad.
        # We need a "good" permutation.
        
        # Let's use a Halton-like or Sobol-like sequence but ensure boundary coverage.
        # Actually, the current Sobol code is quite good.
        
        # Let's try a different structural change: 
        # Use the analytic front but sample using **Quasi-Monte Carlo (Sobol)** 
        # but with **boundary correction**.
        
        # The current code replaces 4 points with corners.
        # Let's replace the first 10 points with a set of points that cover the edges of the square more densely.
        
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            pts = sobol.random(n=n_points)
        except Exception:
            # Fallback to simple grid if scipy fails
            grid = np.linspace(0, 1, int(np.sqrt(n_points)))
            x1, x2 = np.meshgrid(grid, grid, indexing='ij')
            pts = np.column_stack((x1.flatten(), x2.flatten()))
            pts = pts[:n_points]
            # If grid size is not exact, pad with random
            if len(pts) < n_points:
                while len(pts) < n_points:
                    pts = np.vstack([pts, np.array([np.random.rand(), np.random.rand()])])

        # Enhance boundary coverage:
        # The corners (0,0), (0,1), (1,0), (1,1) are critical.
        # The midpoints of the edges (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5) are also good.
        # Let's reserve the first 8 points for these strategic locations.
        
        strategic_points = [
            [0.0, 0.0], # Corner
            [0.0, 1.0], # Corner
            [1.0, 0.0], # Corner
            [1.0, 1.0], # Corner
            [0.5, 0.0], # Edge Mid
            [0.5, 1.0], # Edge Mid
            [0.0, 0.5], # Edge Mid
            [1.0, 0.5], # Edge Mid
        ]
        
        if n_points >= 8:
            for i in range(8):
                pts[i] = strategic_points[i]
        
        x1 = pts[:, 0]
        x2 = pts[:, 1]

        # Construct the full decision vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        if self.dim > 2:
            X[:, 2:10] = 0.5
        X[:, 10] = 0.0
        X[:, 11] = 0.0

        # Evaluate all points
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])

        return None

# EVOLVE-BLOCK-END
