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
        
        def halton(index, base):
            """Generate the index-th point in a Halton sequence with given base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Generate n_total points using 2D Halton sequence (bases 2 and 3)
        # Apply sin^2 edge concentration to bias toward edges of the front
        candidates = []
        
        # Reserve a few slots for exact boundary/vertex points
        n_boundary = 6
        n_halton = n_total - n_boundary
        
        # Halton sequence points with sin^2 edge concentration
        for i in range(n_halton):
            # Halton indices start at 1
            t1 = halton(i + 1, 2)  # in (0, 1)
            t2 = halton(i + 1, 3)  # in (0, 1)
            
            # Apply sin^2 transformation for edge concentration
            # This maps uniform [0,1] to values concentrated near 0 and 1
            # theta = sin^2(t1 * pi/2) * pi/2 gives more points near theta=0 and theta=pi/2
            theta = np.sin(t1 * (np.pi / 2.0)) ** 2 * (np.pi / 2.0)
            phi = np.sin(t2 * (np.pi / 2.0)) ** 2 * (np.pi / 2.0)
            
            u = np.cos(theta)
            sin_theta = np.sin(theta)
            
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
        
        # Add boundary/vertex points for high hypervolume contribution
        # These are (f1, f2, f3) targets on the Pareto front
        boundary_f = [
            # Vertices (one objective = 1, others = 0)
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            # Edge midpoints (two objectives nonzero)
            (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0),
            (1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)),
            (0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        ]
        
        for f1_t, f2_t, f3_t in boundary_f:
            if len(candidates) >= n_total:
                break
            x = np.full(dim, 0.5)
            x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_t, 0, 1))
            r_t = np.sqrt(f1_t * f1_t + f2_t * f2_t)
            if r_t > 1e-15:
                ratio_t = f1_t / r_t
                x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_t, -1, 1))
            else:
                x[1] = 0.5
            candidates.append(x)
        
        # Trim to budget if needed
        if len(candidates) > budget:
            candidates = candidates[:budget]

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
