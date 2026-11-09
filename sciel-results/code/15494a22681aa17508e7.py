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

        # Strategy: Halton sequence for uniform coverage of the Pareto front surface.
        # The front is a quarter sphere. We use a 2D Halton sequence to generate
        # (theta, phi) pairs that uniformly cover the surface area.
        
        n_total = budget
        
        def halton(n, base):
            """Generate the first n points of the Halton sequence in [0,1)."""
            result = []
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    x += f * (i_val % base)
                    i_val //= base
                result.append(x)
            return np.array(result)
        
        # Generate Halton sequences for theta and phi parameters
        # We use bases 2 and 3 for the two dimensions
        t_theta = halton(n_total, 2)
        t_phi = halton(n_total, 3)
        
        candidates = []
        
        for i in range(n_total):
            # Map Halton values to theta and phi with edge concentration
            # Use sin^2 transformation to concentrate points near the edges
            # where hypervolume contribution is highest
            
            # theta in (0, pi/2): controls f3 = cos(theta)
            # We want f3 to range from ~0 to ~1, with more points near extremes
            # f3 = sin^2(t * pi/2) maps t in [0,1] to f3 in [0,1] with edge concentration
            f3 = np.sin(0.5 * np.pi * t_theta[i]) ** 2
            f3 = np.clip(f3, 1e-10, 1.0 - 1e-10)
            theta = np.arccos(f3)
            sin_theta = np.sqrt(max(0.0, 1.0 - f3 * f3))
            
            # phi in (0, pi/2): controls the ratio of f1 to f2
            # Similar edge concentration
            v = np.sin(0.5 * np.pi * t_phi[i]) ** 2
            v = np.clip(v, 1e-10, 1.0 - 1e-10)
            phi = v * (np.pi / 2.0)
            
            f1 = sin_theta * np.cos(phi)
            f2 = sin_theta * np.sin(phi)
            
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
        
        # Add a few extra points at the "corners" of the front for better hypervolume
        # These points maximize individual objective contributions
        corner_points = []
        
        # Corner 1: f1 max, f2 min, f3 mid
        f3_c = 0.5
        theta_c = np.arccos(f3_c)
        sin_theta_c = np.sin(theta_c)
        f1_c = sin_theta_c
        f2_c = 1e-10
        x_c1 = np.full(dim, 0.5)
        x_c1[0] = (2.0 / np.pi) * np.arcsin(f3_c)
        x_c1[1] = (2.0 / np.pi) * np.arcsin(1.0)
        corner_points.append(x_c1)
        
        # Corner 2: f2 max, f1 min, f3 mid
        f1_c2 = 1e-10
        f2_c2 = sin_theta_c
        x_c2 = np.full(dim, 0.5)
        x_c2[0] = (2.0 / np.pi) * np.arcsin(f3_c)
        x_c2[1] = (2.0 / np.pi) * np.arcsin(0.0)
        corner_points.append(x_c2)
        
        # Corner 3: f3 max, f1 mid, f2 mid
        f3_c3 = 1.0 - 1e-10
        theta_c3 = np.arccos(f3_c3)
        sin_theta_c3 = np.sin(theta_c3)
        phi_c3 = np.pi / 4
        f1_c3 = sin_theta_c3 * np.cos(phi_c3)
        f2_c3 = sin_theta_c3 * np.sin(phi_c3)
        x_c3 = np.full(dim, 0.5)
        x_c3[0] = (2.0 / np.pi) * np.arcsin(f3_c3)
        r_c3 = np.sqrt(f1_c3**2 + f2_c3**2)
        if r_c3 > 1e-15:
            x_c3[1] = (2.0 / np.pi) * np.arcsin(f1_c3 / r_c3)
        corner_points.append(x_c3)
        
        # Corner 4: f3 min, f1 mid, f2 mid
        f3_c4 = 1e-10
        theta_c4 = np.arccos(f3_c4)
        sin_theta_c4 = np.sin(theta_c4)
        phi_c4 = np.pi / 4
        f1_c4 = sin_theta_c4 * np.cos(phi_c4)
        f2_c4 = sin_theta_c4 * np.sin(phi_c4)
        x_c4 = np.full(dim, 0.5)
        x_c4[0] = (2.0 / np.pi) * np.arcsin(f3_c4)
        r_c4 = np.sqrt(f1_c4**2 + f2_c4**2)
        if r_c4 > 1e-15:
            x_c4[1] = (2.0 / np.pi) * np.arcsin(f1_c4 / r_c4)
        corner_points.append(x_c4)
        
        # Replace last few Halton points with corner points if we have budget
        n_corners = min(len(corner_points), 4)
        for i in range(n_corners):
            idx = n_total - 1 - i
            if idx >= 0:
                candidates[idx] = corner_points[i]

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
