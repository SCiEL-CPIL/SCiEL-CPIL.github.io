"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Sparse global exploration to find a good starting basin.
    2. Coordinate descent with adaptive step sizes to handle the anisotropy
       of Bent Cigar (one stiff dimension, rest soft).
    3. Powell's method for robust local refinement.
    4. Final high-precision coordinate polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 285.0
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

        def time_ok():
            return state["ok"] and time.perf_counter() < deadline

        # --- Phase 1: Global Exploration ---
        # Use a mix of center, axis-aligned, and Sobol points
        n_global = min(12, max(4, budget // 40))
        
        # Generate candidate points: center, +/- along each axis, and Sobol
        candidates = [np.zeros(d)]
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = sign * 3.0
                candidates.append(pt)
        
        # Add Sobol points
        try:
            from scipy.stats import qmc
            n_sobol = max(1, n_global - len(candidates))
            if n_sobol > 0:
                sampler = qmc.Sobol(d=d, scramble=True, seed=123)
                sobol_pts = sampler.random(n_sobol)
                sobol_pts = -5.0 + 10.0 * sobol_pts
                candidates.extend(sobol_pts.tolist())
        except Exception:
            rng = np.random.default_rng(0)
            for _ in range(max(0, n_global - len(candidates))):
                candidates.append(rng.uniform(-5, 5, d).tolist())

        best_val = np.inf
        best_pt = np.zeros(d)
        for pt in candidates:
            if not time_ok():
                break
            x = np.asarray(pt, dtype=float)
            v = wf(x)
            if v < best_val:
                best_val = v
                best_pt = x.copy()

        x0 = best_pt.copy()

        # --- Phase 2: Adaptive Coordinate Descent ---
        # This handles the anisotropy: try each dimension, adapt step size
        # based on whether the move improved the function.
        remaining = budget - state["count"]
        if remaining > 10 and time_ok():
            x = x0.copy()
            f_curr = wf(x)
            steps = np.ones(d) * 1.0  # initial step sizes
            max_cd_iters = min(200, remaining // 2)
            
            for it in range(max_cd_iters):
                if not time_ok() or (budget - state["count"]) < 5:
                    break
                improved = False
                for i in range(d):
                    if not time_ok() or (budget - state["count"]) < 3:
                        break
                    # Try positive direction
                    x_try = x.copy()
                    x_try[i] += steps[i]
                    v_try = wf(x_try)
                    if v_try < f_curr:
                        x = x_try
                        f_curr = v_try
                        steps[i] *= 1.5
                        improved = True
                    else:
                        # Try negative direction
                        x_try = x.copy()
                        x_try[i] -= steps[i]
                        v_try = wf(x_try)
                        if v_try < f_curr:
                            x = x_try
                            f_curr = v_try
                            steps[i] *= 1.5
                            improved = True
                        else:
                            # Shrink step for this dimension
                            steps[i] *= 0.5
                            # If step is tiny, skip further attempts this dim
                            if steps[i] < 1e-10:
                                steps[i] = 0.0
                    
                    if f_curr < state["best_f"]:
                        state["best_f"] = f_curr
                        state["best_x"] = x.copy()
                
                if not improved:
                    break

            if f_curr < state["best_f"]:
                state["best_f"] = f_curr
                state["best_x"] = x.copy()
            x0 = x.copy()

        # --- Phase 3: Powell Refinement ---
        remaining = budget - state["count"]
        if remaining > 5 and time_ok():
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": min(500, remaining // 3),
                        "maxfev": remaining,
                        "xrtol": 1e-12,
                        "ftol": 1e-12,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
                    x0 = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 4: Final High-Precision Coordinate Polish ---
        remaining = budget - state["count"]
        if remaining > 5 and time_ok():
            x = x0.copy()
            f_curr = wf(x)
            # Very fine coordinate search with small steps
            fine_steps = np.ones(d) * 1e-4
            for _ in range(min(50, remaining // 2)):
                if not time_ok() or (budget - state["count"]) < 3:
                    break
                improved = False
                for i in range(d):
                    if not time_ok() or (budget - state["count"]) < 2:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * fine_steps[i]
                        v_try = wf(x_try)
                        if v_try < f_curr:
                            x = x_try
                            f_curr = v_try
                            improved = True
                            break
                    if f_curr < state["best_f"]:
                        state["best_f"] = f_curr
                        state["best_x"] = x.copy()
                if not improved:
                    # Shrink steps
                    fine_steps *= 0.5
                    if np.all(fine_steps < 1e-14):
                        break

            if f_curr < state["best_f"]:
                state["best_f"] = f_curr
                state["best_x"] = x.copy()

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
