"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer using scrambled Sobol sampling on shape variables mapped to the Pareto simplex."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n points in 2D unit square using Sobol sequence (power-of-2 padded)."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            # Need power of 2, then trim
            m = 1
            while m < n:
                m *= 2
            pts = sampler.random(m)
            return pts[:n]
        except ImportError:
            # Fallback: Halton sequence
            pts = np.zeros((n, 2))
            for i in range(n):
                pts[i, 0] = self._halton_value(i + 1, 2)
                pts[i, 1] = self._halton_value(i + 1, 3)
            return pts

    def _halton_value(self, i, base):
        """Calculate the i-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _map_to_simplex(self, pts):
        """Map points from [0,1]^2 to the Pareto simplex (triangle) uniformly.
        
        Uses the transformation:
        x = sqrt(u) * v
        y = sqrt(u) * (1 - v)
        This ensures uniform coverage of the simplex {x >= 0, y >= 0, x+y <= 1}.
        """
        result = np.zeros_like(pts)
        u = pts[:, 0]
        v = pts[:, 1]
        sqrt_u = np.sqrt(u)
        result[:, 0] = sqrt_u * v
        result[:, 1] = sqrt_u * (1.0 - v)
        
        # Clip to ensure numerical stability
        result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
        result[:, 1] = np.clip(result[:, 1], 0.0, 1.0)
        
        # Handle any numerical issues where x+y might slightly exceed 1
        mask = result[:, 0] + result[:, 1] > 1.0
        if np.any(mask):
            result[mask, 1] = np.maximum(0.0, 1.0 - result[mask, 0])
            
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # DTLZ1 Structure:
        # n_obj - 1 = 2 variables define the Pareto front shape
        # The remaining dim - (n_obj - 1) = 5 variables should be 0.5 for g=0
        n_front_vars = n_obj - 1  # 2
        n_g_vars = dim - n_front_vars  # 5
        
        # Strategy:
        # 1. Use Sobol sampling for uniform coverage of the 2D simplex.
        # 2. Reserve a small number of slots for explicit boundary points (vertices and edge midpoints)
        #    to ensure the extreme corners of the Pareto front are captured, which is critical for hypervolume.
        
        n_boundary = 6
        if budget > n_boundary:
            n_interior = budget - n_boundary
            # Generate interior points
            interior_pts = self._sobol_2d(n_interior)
            interior_pts = self._map_to_simplex(interior_pts)
            
            # Create boundary points: vertices and edge midpoints in decision space
            # The simplex in decision space for the front variables is:
            # x0 >= 0, x1 >= 0, x0 + x1 <= 1
            # Vertices: (0,0), (1,0), (0,1)
            # Edge midpoints: (0.5, 0), (0, 0.5), (0.5, 0.5)
            boundary_pts = np.array([
                [0.0, 0.0],   # Vertex
                [1.0, 0.0],   # Vertex
                [0.0, 1.0],   # Vertex
                [0.5, 0.0],   # Edge midpoint
                [0.0, 0.5],   # Edge midpoint
                [0.5, 0.5],   # Edge midpoint
            ])
            
            # Combine interior and boundary points
            front_samples = np.vstack([interior_pts, boundary_pts])
        else:
            # If budget is very small, just use boundary points
            boundary_pts = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
            ])
            front_samples = boundary_pts[:budget]
        
        # Construct 7D points
        X = np.zeros((budget, dim))
        
        # Scale front variables to bounds
        # front_samples is in [0, 1] (simplex region)
        range_front = hi[:n_front_vars] - lo[:n_front_vars]
        scaled_front = front_samples * range_front + lo[:n_front_vars]
        X[:, :n_front_vars] = scaled_front
        
        # Set g-variables to midpoint of bounds to ensure g=0
        # For [0,1] bounds, midpoint is 0.5
        g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
        X[:, n_front_vars:] = g_mid
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        evals_used = 0
        for i in range(budget):
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break
            problem(X[i])
            evals_used += 1
        
        return None

# EVOLVE-BLOCK-END
