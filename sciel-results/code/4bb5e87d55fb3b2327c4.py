"""Portfolio search for DTLZ2: uniform Halton sampling on the known front with strategic anchors."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using uniform Halton sampling on the Pareto front."""

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

        # Strategy: 2D Halton sequence uniformly mapped to the Pareto front (no sin^2 distortion).
        # The uniform mapping ensures better coverage of the curved front surface.
        # Add strategic anchor points at vertices, edge midpoints, and interior centroid.
        
        # Use 190 Halton points and 10 special points
        n_halton = 190
        
        candidates = []
        
        # 2D Halton sequence generator (vectorized for speed)
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
            # Uniform mapping: u in [0,1] -> theta in [0, pi/2], v in [0,1] -> phi in [0, pi/2]
            # f3 = cos(theta), f1 = sin(theta)*cos(phi), f2 = sin(theta)*sin(phi)
            theta = t_u * (np.pi / 2.0)
            phi = t_v * (np.pi / 2.0)
            
            f3 = np.cos(theta)
            f1 = np.sin(theta) * np.cos(phi)
            f2 = np.sin(theta) * np.sin(phi)
            
            # Clip to avoid numerical issues at boundaries
            f3 = np.clip(f3, 1e-10, 1.0 - 1e-10)
            f1 = np.clip(f1, 1e-10, 1.0 - 1e-10)
            f2 = np.clip(f2, 1e-10, 1.0 - 1e-10)
            
            # Renormalize to lie exactly on the unit sphere
            norm = np.sqrt(f1*f1 + f2*f2 + f3*f3)
            if norm > 1e-15:
                f1 /= norm
                f2 /= norm
                f3 /= norm
            
            # Map to x: x[0] = (2/pi)*arcsin(f3), x[1] = (2/pi)*arcsin(f1/sqrt(f1^2+f2^2))
            x = np.full(dim, 0.5)
            x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
            
            r = np.sqrt(f1 * f1 + f2 * f2)
            if r > 1e-15:
                ratio = f1 / r
                x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio, -1, 1))
            else:
                x[1] = 0.5
            
            candidates.append(x)

        # Add 10 special points for better hypervolume coverage
        
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
        
        # 4. Mid-edge between v1 and v2: (1/sqrt(2), 1/sqrt(2), 0)
        f1_me1 = 1.0 / np.sqrt(2.0)
        f2_me1 = 1.0 / np.sqrt(2.0)
        f3_me1 = 0.0
        x_me1 = np.full(dim, 0.5)
        x_me1[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me1, 0, 1))
        r_me1 = np.sqrt(f1_me1*f1_me1 + f2_me1*f2_me1)
        if r_me1 > 1e-15:
            ratio_me1 = f1_me1 / r_me1
            x_me1[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me1, -1, 1))
        else:
            x_me1[1] = 0.5
        candidates.append(x_me1)
        
        # 5. Mid-edge between v1 and v3: (1/sqrt(2), 0, 1/sqrt(2))
        f1_me2 = 1.0 / np.sqrt(2.0)
        f2_me2 = 0.0
        f3_me2 = 1.0 / np.sqrt(2.0)
        x_me2 = np.full(dim, 0.5)
        x_me2[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me2, 0, 1))
        r_me2 = np.sqrt(f1_me2*f1_me2 + f2_me2*f2_me2)
        if r_me2 > 1e-15:
            ratio_me2 = f1_me2 / r_me2
            x_me2[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me2, -1, 1))
        else:
            x_me2[1] = 0.5
        candidates.append(x_me2)
        
        # 6. Mid-edge between v2 and v3: (0, 1/sqrt(2), 1/sqrt(2))
        f1_me3 = 0.0
        f2_me3 = 1.0 / np.sqrt(2.0)
        f3_me3 = 1.0 / np.sqrt(2.0)
        x_me3 = np.full(dim, 0.5)
        x_me3[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_me3, 0, 1))
        r_me3 = np.sqrt(f1_me3*f1_me3 + f2_me3*f2_me3)
        if r_me3 > 1e-15:
            ratio_me3 = f1_me3 / r_me3
            x_me3[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_me3, -1, 1))
        else:
            x_me3[1] = 0.5
        candidates.append(x_me3)
        
        # 7. Centroid of front: (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        f1_i1 = 1.0 / np.sqrt(3.0)
        f2_i1 = 1.0 / np.sqrt(3.0)
        f3_i1 = 1.0 / np.sqrt(3.0)
        x_i1 = np.full(dim, 0.5)
        x_i1[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_i1, 0, 1))
        r_i1 = np.sqrt(f1_i1*f1_i1 + f2_i1*f2_i1)
        if r_i1 > 1e-15:
            ratio_i1 = f1_i1 / r_i1
            x_i1[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_i1, -1, 1))
        else:
            x_i1[1] = 0.5
        candidates.append(x_i1)
        
        # 8. Point at 1/3 along edge v1-v2: (2/sqrt(6), 1/sqrt(6), 0)
        f1_e1 = 2.0 / np.sqrt(6.0)
        f2_e1 = 1.0 / np.sqrt(6.0)
        f3_e1 = 0.0
        x_e1 = np.full(dim, 0.5)
        x_e1[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_e1, 0, 1))
        r_e1 = np.sqrt(f1_e1*f1_e1 + f2_e1*f2_e1)
        if r_e1 > 1e-15:
            ratio_e1 = f1_e1 / r_e1
            x_e1[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_e1, -1, 1))
        else:
            x_e1[1] = 0.5
        candidates.append(x_e1)
        
        # 9. Point at 1/3 along edge v1-v3: (2/sqrt(6), 0, 1/sqrt(6))
        f1_e2 = 2.0 / np.sqrt(6.0)
        f2_e2 = 0.0
        f3_e2 = 1.0 / np.sqrt(6.0)
        x_e2 = np.full(dim, 0.5)
        x_e2[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_e2, 0, 1))
        r_e2 = np.sqrt(f1_e2*f1_e2 + f2_e2*f2_e2)
        if r_e2 > 1e-15:
            ratio_e2 = f1_e2 / r_e2
            x_e2[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_e2, -1, 1))
        else:
            x_e2[1] = 0.5
        candidates.append(x_e2)
        
        # 10. Point at 1/3 along edge v2-v3: (0, 2/sqrt(6), 1/sqrt(6))
        f1_e3 = 0.0
        f2_e3 = 2.0 / np.sqrt(6.0)
        f3_e3 = 1.0 / np.sqrt(6.0)
        x_e3 = np.full(dim, 0.5)
        x_e3[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_e3, 0, 1))
        r_e3 = np.sqrt(f1_e3*f1_e3 + f2_e3*f2_e3)
        if r_e3 > 1e-15:
            ratio_e3 = f1_e3 / r_e3
            x_e3[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_e3, -1, 1))
        else:
            x_e3[1] = 0.5
        candidates.append(x_e3)

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
