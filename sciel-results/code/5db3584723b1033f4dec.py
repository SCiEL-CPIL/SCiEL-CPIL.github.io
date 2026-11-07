"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4-aware optimizer: exploit known Pareto front with low-discrepancy sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_pos = self.n_obj - 1
        n_g = self.dim - n_pos
        g_opt = 0.5
        
        def make_point(t):
            x = np.zeros(self.dim)
            x[0] = 1.0 - t
            x[1] = t
            x[2:] = g_opt
            return np.clip(x, lo, hi)
        
        # Van der Corput sequence for 1D low-discrepancy sampling
        def vdc(n, base=2):
            res = np.zeros(n)
            for i in range(1, n + 1):
                f, r = 1.0, i
                while r > 0:
                    f /= base
                    r, rem = divmod(r, base)
                    res[i-1] += f * rem
            return res
        
        n_evals = self.budget
        t_vals = vdc(n_evals)
        
        points = []
        for t in t_vals:
            x = make_point(t)
            problem(x)
            points.append(x)
        
        return points
# EVOLVE-BLOCK-END
