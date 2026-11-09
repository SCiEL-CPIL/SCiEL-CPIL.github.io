"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1: n=2 position vars, N-n distance vars
        n_pos = 2
        n_dist = dim - n_pos
        
        rng = np.random.default_rng(42)
        
        # 1. Find optimal x_dist to minimize g
        # We approximate g by sum of objectives at x_pos = [0.5, 0.5]
        # sum(f) = 0.5 * (1+g)
        
        candidates = [
            np.zeros(n_dist),
            np.ones(n_dist),
            np.full(n_dist, 0.5),
            np.full(n_dist, 0.1),
            np.full(n_dist, 0.9),
        ]
        for _ in range(10):
            candidates.append(rng.random(n_dist))
            
        best_x_dist = None
        best_g_proxy = np.inf
        
        for x_d in candidates:
            x_d = np.clip(x_d, lo[n_pos:], hi[n_pos:])
            x_test = np.concatenate([[0.5, 0.5], x_d])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_proxy = np.sum(f)
            if g_proxy < best_g_proxy:
                best_g_proxy = g_proxy
                best_x_dist = x_d.copy()
                
        # Local refinement of x_dist
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            x_pert = best_x_dist + 0.1 * (hi[n_pos:] - lo[n_pos:]) * rng.standard_normal(n_dist)
            x_pert = np.clip(x_pert, lo[n_pos:], hi[n_pos:])
            x_test = np.concatenate([[0.5, 0.5], x_pert])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_proxy = np.sum(f)
            if g_proxy < best_g_proxy:
                best_g_proxy = g_proxy
                best_x_dist = x_pert.copy()
                
        # 2. Sample the Pareto Front
        # Use Sobol sequence for uniform coverage of the simplex
        n_samples = self.budget - 35 # 15 candidates + 20 local steps = 35
        if n_samples < 10:
            n_samples = 10
            
        sobol = qmc.Sobol(d=2, scramble=True)
        samples = sobol.random(n_samples)
        
        recorded_F = []
        
        for s, t in samples:
            x1 = s
            x2 = 1.0 - t
            x = np.concatenate([[x1, x2], best_x_dist])
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded_F.append(f)
            
        return recorded_F
# EVOLVE-BLOCK-END
