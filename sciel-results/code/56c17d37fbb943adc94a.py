"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    CMA-ES based optimizer with multi-start exploration.
    Strategy:
    1. Generate diverse starting points.
    2. Run short CMA-ES from each start to find local minima.
    3. Use remaining budget for a final CMA-ES polish on the global best.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Multi-start CMA-ES with diverse initializations
        # Use a smaller population and shorter runs for each start
        n_starts = 8
        pop_size = 12  # Small population for CMA-ES
        max_iter_per_start = 30

        # Generate starting points
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        for s in starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            s = np.clip(s, -5, 5)

            try:
                res = minimize(
                    wrapped,
                    s,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter_per_start, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

        # Phase 2: Final CMA-ES polish on the global best
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(150, max(20, remaining // 3))
                
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
