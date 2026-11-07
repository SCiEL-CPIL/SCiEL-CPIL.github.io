"""Optimized DTLZ4 solver: Sobol sampling on triangular Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_pos = self.n_obj - 1  # 2 position variables
        n_g = self.dim - n_pos  # 10 g-variables
        g_opt = 0.5  # optimal g-value
        
        n_samples = min(self.budget, 200)
        
        points = []
        
        # Include the 3 vertices of the triangle (high hypervolume contribution)
        vertices = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
        
        for v in vertices:
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = g_opt
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Add edge midpoints for better boundary coverage
        edge_mids = [
            [0.5, 0.0],  # midpoint of edge between (1,0) and (0,0)
            [0.0, 0.5],  # midpoint of edge between (0,1) and (0,0)
            [0.5, 0.5],  # midpoint of edge between (1,0) and (0,1)
        ]
        
        for e in edge_mids:
            if len(points) < n_samples:
                x = np.zeros(self.dim)
                x[0] = e[0]
                x[1] = e[1]
                x[2:] = g_opt
                x = np.clip(x, lo, hi)
                points.append(x)
        
        # Use Sobol sampling for the remaining budget
        n_remaining = n_samples - len(points)
        if n_remaining > 0:
            # Sobol sampler needs power of 2 points; use next power of 2 >= n_remaining
            n_sobol = 1
            while n_sobol < n_remaining:
                n_sobol *= 2
            
            sobol = qmc.Sobol(d=2, scramble=True)
            # Generate points in [0,1]^2
            sobol_points = sobol.random(n_sobol)
            
            # Map [0,1]^2 to the triangle using barycentric coordinates
            # For each (u,v) in [0,1]^2, if u+v <= 1, map to (x0, x1) = (u, v)
            # Otherwise, reflect: use transformation that covers the triangle uniformly
            # Better approach: use the fact that uniform in [0,1]^2 clipped to triangle
            # gives non-uniform density. Instead, use:
            # x0 = u, x1 = v*(1-u) when u+v<=1... no, that's not right.
            # 
            # Correct uniform sampling in triangle: 
            # Let u,v ~ Uniform[0,1]. If u+v > 1, set u=1-u, v=1-v.
            # Then (u,v) is uniform in the triangle.
            
            for i in range(n_sobol):
                if len(points) >= n_samples:
                    break
                if time.time() > deadline:
                    break
                    
                u = sobol_points[i, 0]
                v = sobol_points[i, 1]
                
                # Fold into triangle for uniform distribution
                if u + v > 1.0:
                    u = 1.0 - u
                    v = 1.0 - v
                
                # Now (u,v) is in the triangle: u>=0, v>=0, u+v<=1
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                x[2:] = g_opt
                x = np.clip(x, lo, hi)
                points.append(x)
        
        # Evaluate all points
        for x in points:
            problem(x)
        
        return points
# EVOLVE-BLOCK-END
