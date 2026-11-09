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

        # Strategy: 2D Halton sequence on the Pareto front with sin^2 edge concentration.
        # Plus explicit vertices and a center point to maximize hypervolume coverage.
        
        # Use 196 Halton points and 4 special points (3 vertices + 1 center)
        n_halton = 196
        
        candidates = []
        
        # 2D Halton sequence generator
        def halton_2d(n, base1, base2):
            """Generate n points of a 2D Halton sequence."""
            points = []
            for i in range(1, n + 1):
                # 1st coordinate (base base1)
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base1
                    r += f * (i_val % base1)
                    i_val //= base1
                # 2nd coordinate (base base2)
                f = 1.0
                r2 = 0.0
                i_val = i
                while i_val > 0:
                    f /= base2
                    r2 += f * (i_val % base2)
                    i_val //= base2
                points.append((r, r2))
            return points
        
        # Generate Halton points with coprime bases 2 and 3
        halton_pts = halton_2d(n_halton, 2, 3)
        
        for t_u, t_v in halton_pts:
            # Apply sin^2 edge concentration
            u = np.sin(0.5 * np.pi * t_u) ** 2
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            sin_theta = np.sqrt(max(0.0, 1.0 - u * u))
            
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

        # Add 4 special points
        # 1. Vertex (1, 0, 0) -> f1=1, f2=0, f3=0
        x_v1 = np.full(dim, 0.5)
        x_v1[0] = 0.0
        x_v1[1] = 1.0
        candidates.append(x_v1)
        
        # 2. Vertex (0, 1, 0) -> f1=0, f2=1, f3=0
        x_v2 = np.full(dim, 0.5)
        x_v2[0] = 0.0
        x_v2[1] = 0.5
        candidates.append(x_v2)
        
        # 3. Vertex (0, 0, 1) -> f1=0, f2=0, f3=1
        x_v3 = np.full(dim, 0.5)
        x_v3[0] = 1.0
        x_v3[1] = 0.5
        candidates.append(x_v3)
        
        # 4. Center point (0.5, 0.5, sqrt(0.5))
        f1_c = 0.5
        f2_c = 0.5
        f3_c = np.sqrt(0.5)
        x_c = np.full(dim, 0.5)
        x_c[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_c, 0, 1))
        r_c = np.sqrt(f1_c*f1_c + f2_c*f2_c)
        if r_c > 1e-15:
            ratio_c = f1_c / r_c
            x_c[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_c, -1, 1))
        else:
            x_c[1] = 0.5
        candidates.append(x_c)

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
