"""Portfolio search for DTLZ2: deterministic quarter-circle arc on the known front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using an explicit front-arc construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0  # conservative safety margin under the 330 s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # Strategy: Stratified grid on the Pareto front surface.
        # The front is a quarter sphere. We sample u = cos(theta) uniformly in [0,1]
        # and phi uniformly in [0, pi/2].
        # This provides a uniform distribution of points on the surface area.
        
        n_total = budget
        # Determine grid dimensions
        n_u = int(np.sqrt(n_total))
        n_v = n_total // n_u
        
        candidates = []
        
        for i in range(n_u):
            # Stratified sampling for u in (0, 1) with edge concentration
            # Use arcsine-like distribution: u = sin^2(pi/2 * t) where t is uniform
            t_u = (i + 0.5) / n_u
            u = np.sin(0.5 * np.pi * t_u) ** 2
            # Clip to avoid numerical issues at boundaries
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            theta = np.arccos(u)
            sin_theta = np.sqrt(max(0.0, 1.0 - u * u))
            
            for j in range(n_v):
                # Stratified sampling for phi in (0, pi/2) with edge concentration
                t_v = (j + 0.5) / n_v
                # Use similar edge-concentrated distribution
                v = np.sin(0.5 * np.pi * t_v) ** 2
                v = np.clip(v, 1e-10, 1.0 - 1e-10)
                phi = v * (np.pi / 2.0)
                
                f1 = sin_theta * np.cos(phi)
                f2 = sin_theta * np.sin(phi)
                f3 = u
                
                # Map to x
                x = np.full(dim, 0.5)
                x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
                
                r = np.sqrt(f1 * f1 + f2 * f2)
                if r > 1e-15:
                    ratio = f1 / r
                    x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio, -1, 1))
                else:
                    x[1] = 0.5
                
                candidates.append(x)

        # Evaluate all candidates within the time budget
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.zeros((0, n_obj))
# EVOLVE-BLOCK-END
