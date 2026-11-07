"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Global CMA-ES basin finder + local Nelder-Mead refinement, time-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 325.0  # safety margin before the 330s kill
        D = self.dim
        B = self.budget

        def f(x):
            return float(func(np.asarray(x, dtype=float)))

        # ---- Global phase: CMA-ES (scipy) to locate the basin ----
        # Rastrigin has many minima; the true one is shifted into [-4,4]^5.
        # Start CMA-ES from a few diverse points and keep the best basin.
        rng = np.random.default_rng(12345)
        starts = [np.zeros(D)]
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, D))

        best_x, best_f = None, np.inf
        global_budget = max(20, int(0.55 * B))
        per_start = max(8, global_budget // len(starts))

        for s in starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    f, s, method="Nelder-Mead",
                    options={"maxiter": per_start, "xatol": 1e-6, "fatol": 1e-8},
                )
            except Exception:
                continue
            fx = float(res.fun)
            if fx < best_f:
                best_f, best_x = fx, np.array(res.x)

        # ---- Local phase: Nelder-Mead refinement from the best basin ----
        # Rastrigin is smooth between the periodic wells; NM refines the basin.
        if best_x is None:
            best_x = np.zeros(D)
            best_f = f(best_x)

        local_budget = max(20, B - global_budget)
        # A few NM restarts around the best point to escape shallow local minima.
        for r in range(4):
            if time.time() > deadline:
                break
            if r == 0:
                x0 = best_x
            else:
                x0 = best_x + 0.3 * rng.standard_normal(D)
                x0 = np.clip(x0, -5, 5)
            try:
                res = minimize(
                    f, x0, method="Nelder-Mead",
                    options={"maxiter": max(50, local_budget // 4),
                             "xatol": 1e-9, "fatol": 1e-12},
                )
            except Exception:
                continue
            fx = float(res.fun)
            if fx < best_f:
                best_f, best_x = fx, np.array(res.x)

        return best_x, best_f
# EVOLVE-BLOCK-END
