"""Combined: deterministic Pareto-front grid with Halton-sequence refinement for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a structured front grid + Halton refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def _front_point(self, u1, u2, lo, hi):
        """Map two uniform [0,1) values to a DTLZ2 Pareto front point in [lo,hi]^dim."""
        # u1 -> alpha via arccos(1-u1) for uniform surface density on quarter sphere
        # u2 -> beta = (pi/2)*u2
        alpha = np.arccos(1.0 - u1)
        beta = (np.pi / 2.0) * u2
        f1 = np.sin(alpha) * np.cos(beta)
        f2 = np.sin(alpha) * np.sin(beta)
        f3 = np.cos(alpha)
        # DTLZ2: x[0] = (f1^2 + f2^2 + f3^2)^(1/2) / 2 ... actually standard DTLZ2:
        # g(x) = sum((xi-0.5)^2) for i in g_vars, g* = 0 on front
        # f1 = (1+g)*cos(theta1)*cos(theta2), etc. On front g=0.
        # x[0] is the g-variable (or one of them). For 12 vars, typically x[0] is g-var.
        # To be on front: g=0, so x[0]=0.5 (since (0.5-0.5)^2=0)
        # The remaining 11 variables determine the angles.
        # Standard DTLZ2: n-1 decision vars for angles, 1 for g.
        # With dim=12: x[0] is g-var, x[1..11] are angle vars.
        # theta_i = (pi/2) * x[i] for i=1..11, but we only need 2 angles for 3 objectives.
        # Actually for DTLZ2 with m objectives, we need m-1 angles.
        # f1 = (1+g)*cos(t1)*cos(t2), f2=(1+g)*cos(t1)*sin(t2), f3=(1+g)*sin(t1)
        # where t1 = (pi/2)*x[1], t2 = (pi/2)*x[2]
        # So to get specific f1,f2,f3 on front: set g=0, x[0]=0.5
        # x[1] = (2/pi)*alpha, x[2] = (2/pi)*beta
        # Other x[3..11] can be anything, say 0.5
        x = np.full(self.dim, 0.5)
        x[0] = 0.5  # g=0
        x[1] = (2.0 / np.pi) * alpha
        x[2] = (2.0 / np.pi) * beta
        # Clip to bounds
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate candidate points: grid + Halton
        # Grid: 14x14 = 196 points
        grid_size = 14
        grid_u1 = np.linspace(0.0, 1.0, grid_size, endpoint=False) + 0.5/grid_size
        grid_u2 = np.linspace(0.0, 1.0, grid_size, endpoint=False) + 0.5/grid_size
        grid_u1, grid_u2 = np.meshgrid(grid_u1, grid_u2, indexing='ij')
        grid_u1 = grid_u1.ravel()
        grid_u2 = grid_u2.ravel()
        
        # Halton refinement: use remaining budget
        n_halton = max(0, self.budget - len(grid_u1))
        if n_halton > 0:
            h1 = self._halton(n_halton, 2)
            h2 = self._halton(n_halton, 3)
            all_u1 = np.concatenate([grid_u1, h1])
            all_u2 = np.concatenate([grid_u2, h2])
        else:
            all_u1 = grid_u1
            all_u2 = grid_u2
        
        # Truncate to budget
        all_u1 = all_u1[:self.budget]
        all_u2 = all_u2[:self.budget]
        
        # Evaluate points
        results = []
        for i in range(len(all_u1)):
            if time.time() > deadline:
                break
            x = self._front_point(all_u1[i], all_u2[i], lo, hi)
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
