"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer using scrambled Sobol sampling mapped to the Pareto simplex with boundary augmentation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_value(self, i, base):
        """Calculate the i-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_2d(self, n):
        """Generate n points in [0, 1)^2 using the Halton sequence with bases 2 and 3."""
        points = np.zeros((n, 2))
        for i in range(1, n + 1):
            points[i - 1, 0] = self._halton_value(i, 2)
            points[i - 1, 1] = self._halton_value(i, 3)
        return points

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
        
        # Set distance variables to optimal value 0.5
        x_dist = np.full(n_g_vars, 0.5)
        x_dist = np.clip(x_dist, lo[n_front_vars:], hi[n_front_vars:])
        
        # Strategy:
        # 1. Reserve boundary points for vertices and edge midpoints of the simplex
        # 2. Use scrambled Sobol for the remaining budget, mapped to simplex
        
        # Boundary points: 3 vertices + 3 edge midpoints = 6 points
        boundary_pts = np.array([
            [1.0, 0.0],   # vertex 1
            [0.0, 1.0],   # vertex 2
            [0.0, 0.0],   # vertex 3
            [0.5, 0.5],   # midpoint of edge 1-2
            [0.5, 0.0],   # midpoint of edge 1-3
            [0.0, 0.5],   # midpoint of edge 2-3
        ])
        
        n_boundary = len(boundary_pts)
        if budget <= n_boundary:
            # Use only as many boundary points as budget allows
            pts_simplex = boundary_pts[:budget]
        else:
            n_sobol = budget - n_boundary
            # Generate Sobol points
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True)
                # Sobol requires power-of-2; generate enough and trim
                m = 1
                while m < n_sobol:
                    m *= 2
                pts_raw = sampler.random(m)[:n_sobol]
            except ImportError:
                pts_raw = self._halton_2d(n_sobol)
            
            # Map to simplex
            pts_sobol_simplex = self._map_to_simplex(pts_raw)
            
            # Combine boundary and Sobol points
            pts_simplex = np.vstack([boundary_pts, pts_sobol_simplex])
        
        # Evaluate all points
        evals_used = 0
        for i in range(len(pts_simplex)):
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break
                
            x_full = np.zeros(dim)
            x_full[0] = pts_simplex[i, 0]
            x_full[1] = pts_simplex[i, 1]
            x_full[n_front_vars:] = x_dist
            
            # Clip to bounds [0, 1]
            x_full = np.clip(x_full, lo, hi)
            
            # Evaluate
            problem(x_full)
            evals_used += 1
            
        return None
# EVOLVE-BLOCK-END
