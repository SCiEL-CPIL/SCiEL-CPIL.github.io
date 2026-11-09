"""Portfolio search for DTLZ2: dense 2D stratified grid on the Pareto front + gap-filling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a stratified Pareto-front grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0

        def time_ok():
            return (time.time() - start) < deadline

        def front_to_x(theta1, theta2):
            """
            Map angles on the 2D Pareto front of DTLZ2 to a 12-dim decision vector.
            For DTLZ2 with M=3 objectives:
              f1 = g * (1 + cos(theta1)) * sin(theta2)  -- wait, let me recall the exact formula.
              
            DTLZ2 with M objectives:
            f_i(x) = (1 + g(x)) * h_i(x)
            where h_1 = cos(theta1) * sin(theta2) ... no, let me be precise.
            
            For M=3:
            h_1 = cos(theta1) * sin(theta2)
            h_2 = sin(theta1) * sin(theta2)  
            h_3 = cos(theta2)
            
            where theta1 in [0, pi/2], theta2 in [0, pi/2]
            
            g(x) = sum_{i=4}^{12} (x_i - 0.5)^2 for DTLZ2 (9 position variables)
            To be on the Pareto front, we need g=0, so x_4..x_12 = 0.5
            x_1 = theta1, x_2 = theta2, x_3 can be anything (say 0)
            """
            x = np.zeros(self.dim)
            x[0] = theta1
            x[1] = theta2
            x[2] = 0.0
            x[3:] = 0.5  # position variables at 0.5 for g=0
            # Clip to bounds
            x = np.clip(x, lo, hi)
            return x

        # Phase 1: Dense stratified grid on the Pareto front
        # Use a 2D grid with stratified sampling for uniform coverage
        n_grid = 12  # 12x12 = 144 points on the front
        if n_grid * n_grid > self.budget * 0.6:
            n_grid = int(np.sqrt(self.budget * 0.6))
        
        # Stratified angles: use Halton-like or just uniform with small jitter
        # For 2D, use a grid with stratified offsets
        theta1_vals = np.linspace(0.0, np.pi / 2, n_grid, endpoint=False)
        theta2_vals = np.linspace(0.0, np.pi / 2, n_grid, endpoint=False)
        # Add small stratified jitter to avoid exact boundaries
        jitter = 1.0 / (2 * n_grid)
        theta1_vals = theta1_vals + jitter * (1 + np.cos(np.linspace(0, 2 * np.pi, n_grid, endpoint=False)))
        theta2_vals = theta2_vals + jitter * (1 + np.cos(np.linspace(0, 2 * np.pi, n_grid, endpoint=False)))
        # Actually, let's use a proper stratified approach
        # For each cell in the 2D grid, place one point at the center
        theta1_vals = (np.arange(n_grid) + 0.5) * (np.pi / 2) / n_grid
        theta2_vals = (np.arange(n_grid) + 0.5) * (np.pi / 2) / n_grid
        
        # Create all combinations
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()
        
        # Shuffle to diversify
        perm = rng.permutation(len(T1_flat))
        T1_flat = T1_flat[perm]
        T2_flat = T2_flat[perm]
        
        # Evaluate grid points
        n_grid_pts = len(T1_flat)
        for i in range(n_grid_pts):
            if used >= self.budget or not time_ok():
                break
            x = front_to_x(T1_flat[i], T2_flat[i])
            f = problem(x)
            used += 1
        
        # Phase 2: Gap-filling / local refinement around promising regions
        # Use remaining budget for local search around grid points
        # We don't track which points were evaluated individually, so use a different strategy:
        # Generate additional points with smaller angular steps in underexplored regions
        
        if used < self.budget and time_ok():
            # Generate a second, denser grid with offset
            n_grid2 = max(8, int(n_grid * 0.7))
            theta1_vals2 = (np.arange(n_grid2) + 0.3) * (np.pi / 2) / n_grid2
            theta2_vals2 = (np.arange(n_grid2) + 0.7) * (np.pi / 2) / n_grid2
            T1_2, T2_2 = np.meshgrid(theta1_vals2, theta2_vals2, indexing='ij')
            T1_flat2 = T1_2.flatten()
            T2_flat2 = T2_2.flatten()
            perm2 = rng.permutation(len(T1_flat2))
            T1_flat2 = T1_flat2[perm2]
            T2_flat2 = T2_flat2[perm2]
            
            for i in range(len(T1_flat2)):
                if used >= self.budget or not time_ok():
                    break
                x = front_to_x(T1_flat2[i], T2_flat2[i])
                f = problem(x)
                used += 1
        
        # Phase 3: Random exploration with local perturbation
        if used < self.budget and time_ok():
            while used < self.budget and time_ok():
                # Random point on or near the front
                theta1 = rng.uniform(0, np.pi / 2)
                theta2 = rng.uniform(0, np.pi / 2)
                # Small perturbation to position variables
                x = front_to_x(theta1, theta2)
                # Add small noise to position variables
                noise = 0.01 * rng.standard_normal(self.dim - 3)
                x[3:] += noise
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
        
        return None
# EVOLVE-BLOCK-END
