"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using Sobol points to find a good starting basin.
    2. Local refinement using Nelder-Mead (derivative-free, adaptive simplex shape)
       which stretches along the Bent Cigar valley and converges fast in early/mid
       iterations, improving AOCC.
    3. A final high-precision polish pass with a tight tolerance to squeeze out
       the last bits of precision.
    4. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit (killed at 330s)
        d = self.dim
        budget = self.budget

        state = {"count": 0, "ok": True, "best_x": None, "best_f": np.inf}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300

            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300

            try:
                v = float(func(x))
            except Exception:
                return 1e300

            state["count"] += 1
            if not np.isfinite(v):
                return 1e300

            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()

            return v

        # --- Phase 1: Global Exploration (Sobol) ---
        # A small number of deterministic Sobol points to cover the space cheaply.
        # Bent Cigar is unimodal but shifted/rotated, so finding the basin is key.
        n_sobol = min(20, budget // 10)

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = -5.0 + 10.0 * sobol_points

            best_idx = 0
            best_val = np.inf

            for i in range(n_sobol):
                if not state["ok"]:
                    break
                x = sobol_points[i]
                v = wf(x)
                if v < best_val:
                    best_val = v
                    best_idx = i

            x0 = sobol_points[best_idx].copy()
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Nelder-Mead) ---
        # Nelder-Mead adapts the simplex shape to the anisotropy, stretching along
        # the Bent Cigar valley more effectively than isotropic methods, giving
        # faster early/mid convergence and better AOCC.
        remaining_budget = budget - state["count"]

        if remaining_budget > 0 and state["ok"]:
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 2000,
                        "maxfev": remaining_budget,
                        "xatol": 1e-10,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 3: High-precision polish pass ---
        # A tight-tolerance Nelder-Mead run in a small neighborhood around the
        # best point to squeeze out the last bits of precision.
        remaining_budget = budget - state["count"]
        if remaining_budget > 5 and state["ok"] and state["best_x"] is not None:
            try:
                x_start = np.clip(state["best_x"], -5.0, 5.0)
                res = minimize(
                    wf,
                    x_start,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xatol": 1e-12,
                        "fatol": 1e-12,
                        "initial_simplex": np.vstack([
                            x_start + 1e-3 * np.eye(d),
                            x_start,
                        ]),
                        "adaptive": True,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
