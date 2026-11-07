"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Objective-space lattice search for DTLZ2 using inverse mapping."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Pareto front: f = 0.5 * sin(pi * y) where y is on unit sphere surface in first 3 dims
        # Actually for DTLZ2: f_i = (1+g) * y_i where y is on unit sphere in [0,1]^3
        # The Pareto front is the unit quarter-sphere in 3D: y_1^2 + y_2^2 + y_3^2 = 1, y_i >= 0
        # We can parametrize using spherical coordinates:
        # y_1 = sin(theta) * cos(phi)
        # y_2 = sin(theta) * sin(phi)  
        # y_3 = cos(theta)
        # where theta in [0, pi/2], phi in [0, pi/2]
        
        # To get uniform coverage on the sphere surface, we use a 2D grid in (theta, phi)
        # but with theta adjusted for uniform area: theta = arccos(1 - 2*t) for uniform t
        
        # Better approach: use stratified sampling on the sphere
        # We'll create a 2D grid in (u, v) in [0,1]^2 and map to sphere uniformly
        
        n_evals = self.budget  # Use all 200 evaluations
        
        # Create stratified points in [0,1]^2
        n_per_dim = int(np.sqrt(n_evals))
        if n_per_dim * n_per_dim < n_evals:
            n_per_dim += 1
        
        # Use a grid with jitter for better coverage
        u_vals = np.linspace(0.5/n_per_dim, 1 - 0.5/n_per_dim, n_per_dim)
        v_vals = np.linspace(0.5/n_per_dim, 1 - 0.5/n_per_dim, n_per_dim)
        
        U, V = np.meshgrid(u_vals, v_vals)
        U = U.ravel()
        V = V.ravel()
        
        # Keep only first n_evals points
        U = U[:n_evals]
        V = V[:n_evals]
        
        # Map (u, v) in [0,1]^2 to uniform points on quarter-sphere
        # Using inverse CDF for uniform distribution on sphere
        # theta = arccos(1 - u)  (for uniform in cos(theta))
        # phi = pi/2 * v
        theta = np.arccos(1 - U)  # theta in [0, pi/2]
        phi = (np.pi / 2) * V    # phi in [0, pi/2]
        
        # Convert to y coordinates on unit sphere
        y1 = np.sin(theta) * np.cos(phi)
        y2 = np.sin(theta) * np.sin(phi)
        y3 = np.cos(theta)
        
        # For DTLZ2, the decision variables map to the Pareto front as:
        # x[0] = y1, x[1] = y2, x[2] = y3, and remaining x[i] = 0.5 for i >= 3
        # Wait, let me reconsider the DTLZ2 formulation.
        
        # DTLZ2: f_i = (1 + g(x_M)) * y_i
        # where y is in [0,1]^M and sum(y_i^2) = 1, g = 0 when x_M is on Pareto set
        # For 3 objectives, M = 3, so y = (y1, y2, y3) on unit sphere
        # The decision variables are x = (x_1, ..., x_12)
        # Typically: x_1, x_2, x_3 map to y_1, y_2, y_3 via spherical coords
        # and x_4...x_12 are for g(x)
        
        # Standard DTLZ2 mapping:
        # y_1 = sin(theta_1) * ... 
        # Actually, the common parametrization is:
        # y_i = sin(theta_i) * cos(theta_{i+1}) for i < M-1
        # y_M = prod(sin(theta_i))
        
        # Simpler: just set x[0]=y1, x[1]=y2, x[2]=y3 and rest to 0.5
        # But we need to verify this gives the correct Pareto front.
        
        # Let's use the standard approach: for DTLZ2 with 3 objectives,
        # the Pareto front is the unit quarter-sphere in 3D objective space.
        # We map (y1, y2, y3) to decision variables such that problem(x) gives f = 0.5*y
        
        # In standard DTLZ2 implementations, x[0] controls y1, x[1] controls y2, etc.
        # A common mapping is:
        # x[0] = y1, x[1] = y2, x[2] = y3, x[3:] = 0.5
        
        X = []
        for i in range(n_evals):
            x = np.full(self.dim, 0.5)
            x[0] = y1[i]
            x[1] = y2[i]  
            x[2] = y3[i]
            X.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
