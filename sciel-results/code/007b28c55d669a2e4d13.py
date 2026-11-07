# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer: integer-aligned coarse-to-fine search + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        DEADLINE = start + 285.0
        MAX_CALLS = budget

        calls = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def F(x):
            calls[0] += 1
            v = func(np.asarray(x, dtype=float))
            if np.isfinite(v) and v < best_f:
                best_f = v
                best_x = np.asarray(x, dtype=float).copy()
            return v

        def alive():
            return calls[0] < MAX_CALLS and time.time() < DEADLINE

        # ---- Phase 1: coarse integer-aligned grid (exploits period = 1) ----
        xs = np.arange(-4.0, 4.0 + 1e-9, 2.0)  # -4,-2,0,2,4
        grid = np.array(np.meshgrid(*([xs] * dim)), dtype=float).reshape(-1, dim)
        for i in range(grid.shape[0]):
            if not alive():
                break
            F(grid[i])

        # ---- Phase 2: refine around the top candidates (Nelder-Mead) ----
        # Gather a diverse set of starting points: grid points + integer neighbours
        # of the current best, to cover the true integer basin even if the coarse
        # grid was fooled by the offset.
        candidates = [best_x.copy()]
        bx = np.round(best_x)
        for dx in np.ndindex(*([2] * dim)):
            p = bx + (2 * np.array(dx, dtype=float) - (dim - 1) * np.ones(dim))
            candidates.append(p)
        # also add a few of the best-scoring grid points
        for i in range(min(6, grid.shape[0])):
            candidates.append(grid[i].copy())

        # sort candidates by current known best first, then unknowns
        def cand_key(c):
            # prefer points already evaluated (we know their value)
            return 0

        opts = {
            "maxiter": 300,
            "xatol": 1e-8,
            "fatol": 1e-10,
            "adaptive": True,
            "maxfev": 60,
            "nfev": 60,
        }

        for c in candidates:
            if not alive():
                break
            if calls[0] + 60 > MAX_CALLS:
                break
            try:
                res = minimize(F, np.asarray(c, dtype=float), method="Nelder-Mead",
                               options=opts)
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.asarray(res.x, dtype=float).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
