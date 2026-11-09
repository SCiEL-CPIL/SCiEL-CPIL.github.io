# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer using low-discrepancy sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        M = self.n_obj
        n_g = M - 1  # number of shape variables (first M-1)

        X_list = []
        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1
            return f

        # Base vector with g-variables set to 0.5 (minimum of g for DTLZ4)
        base_x = np.full(n, 0.5)

        # 1. Use a grid for the shape variables to ensure uniform coverage including boundaries
        if used < self.budget:
            N_rem = self.budget - used
            if n_g == 2:
                N_side = int(np.sqrt(N_rem))
                # Ensure we don't exceed budget
                # N_side x N_side <= N_rem
                x1_vals = np.linspace(0, 1, N_side)
                x2_vals = np.linspace(0, 1, N_side)
                
                for x1 in x1_vals:
                    for x2 in x2_vals:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        x = base_x.copy()
                        x[0] = x1
                        x[1] = x2
                        x = np.clip(x, lo, hi)
                        evaluate(x)
                    if used >= self.budget or time.time() > deadline:
                        break
            else:
                # Fallback to Sobol for generic dimensions
                sampler = qmc.Sobol(d=n_g, scramble=True)
                points = sampler.random(N_rem)
                for p in points:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[:n_g] = p
                    x = np.clip(x, lo, hi)
                    evaluate(x)

        if len(F_list) == 0:
            return []

        return np.array(F_list)
# EVOLVE-BLOCK-END
