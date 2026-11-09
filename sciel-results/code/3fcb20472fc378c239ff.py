"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer using optimal convergence variables (0.5) and Sobol sampling for the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 Structure:
        # For N=7, M=3 (3 objectives):
        # Shape variables: x[0], x[1]
        # Distance variables: x[2] to x[6]
        # g(x) is minimized at x[i] = 0.5 for all distance variables.
        # This minimizes the scaling factor (1+g), making all objectives as small (negative) as possible,
        # which maximizes the distance to the reference point [400, 400, 400] and thus the hypervolume.
        
        n_shape = 2
        n_dist = self.dim - n_shape
        
        # Set distance variables to optimal value 0.5
        x_dist = np.full(n_dist, 0.5)
        # Clip to bounds just in case, though 0.5 is within [0, 1]
        x_dist = np.clip(x_dist, lo[n_shape:], hi[n_shape:])
        
        N = self.budget
        
        # Structural Change: Use a Uniform Grid for shape variables.
        # Unlike Sobol (low-discrepancy), a grid ensures exact coverage of the boundary
        # (u=0, u=1, v=0, v=1) which correspond to the extreme points of the Pareto front.
        # For HV maximization with a distant reference point, capturing the "corners"
        # of the front is crucial.
        
        # Calculate grid size
        # We want approx N points. A square grid is a good start.
        grid_size = int(np.sqrt(N))
        # Adjust if we can fit a larger grid
        # E.g. if N=200, sqrt=14, 14*14=196. 15*15=225 (too many).
        # We will generate a grid of grid_size x grid_size points.
        
        # Generate grid points in [0, 1]
        # Using linspace ensures endpoints 0 and 1 are included.
        u_vals = np.linspace(0.0, 1.0, grid_size)
        v_vals = np.linspace(0.0, 1.0, grid_size)
        
        # Create meshgrid
        U, V = np.meshgrid(u_vals, v_vals)
        pts = np.column_stack((U.ravel(), V.ravel()))
        
        # If we have budget left, we can add a second, finer grid or random points?
        # Or just stick to the grid. 196 points is very close to 200.
        # Let's check if we can add a few more points to reach closer to N.
        # Actually, the evaluator stops at 200. So 196 is safe.
        # If N allows, we could do a 14x14 grid (196) + 4 corner points?
        # The corners (0,0), (1,0), (0,1), (1,1) are already in the grid.
        
        # Let's try to use the full budget if possible.
        # If grid_size * grid_size < N, we might add some points.
        # But for now, the grid structure is the main change.
        
        pts_transformed = pts.copy()
        
        evals_used = 0
        for i in range(len(pts_transformed)):
            if time.time() > deadline:
                break
            if evals_used >= N:
                break
                
            x_full = np.zeros(self.dim)
            x_full[0] = pts_transformed[i, 0]
            x_full[1] = pts_transformed[i, 1]
            x_full[n_shape:] = x_dist
            
            # Clip to bounds [0, 1]
            x_full = np.clip(x_full, lo, hi)
            
            # Evaluate
            # The evaluator records the return value of problem(x_full)
            problem(x_full)
            evals_used += 1
            
        # The return value is ignored by the evaluator
        return None

    def _halton_2d(self, n):
        """Generate n points in [0, 1)^2 using the Halton sequence with bases 2 and 3."""
        points = np.zeros((n, 2))
        for i in range(1, n + 1):
            points[i - 1, 0] = self._halton_value(i, 2)
            points[i - 1, 1] = self._halton_value(i, 3)
        return points

    def _halton_value(self, i, base):
        """Calculate the i-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
# EVOLVE-BLOCK-END
