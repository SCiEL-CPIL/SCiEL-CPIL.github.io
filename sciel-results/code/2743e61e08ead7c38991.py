"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 315.0  # evaluator kills at ~330s; keep a safety margin


class Optimizer:
    """(1+1)-ES with window-based success-rate sigma adaptation,
    multi-point initialization, aggressive stall restarts, and
    adaptive coordinate-descent polish. Minimizes func on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        eval_count = 0

        def _time_ok():
            return time.perf_counter() - t0 < _DEADLINE

        def _safe_call(x):
            nonlocal eval_count
            if eval_count >= budget or not _time_ok():
                return None
            fx = func(x)
            eval_count += 1
            return fx

        rng = np.random.default_rng(42)

        # ---- Multi-point initialization: evaluate several diverse starts ----
        # Schwefel f20 has many local minima; starting from multiple points
        # gives better early progress (lower b_i for small i).
        n_starts = 5
        candidates = []
        # One near origin, others spread across the box
        starts = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]
        for s in starts:
            s = np.clip(s, -5.0, 5.0)
            fs = _safe_call(s)
            if fs is None:
                break
            candidates.append((s, fs))

        if not candidates:
            # Fallback: at least one call was made
            return candidates[0][0], candidates[0][1] if candidates else (np.zeros(dim), 0.0)

        # Pick the best start
        best_idx = min(range(len(candidates)), key=lambda i: candidates[i][1])
        x = candidates[best_idx][0].copy()
        fx = candidates[best_idx][1]

        # ---- Phase 1: (1+1)-ES with window-based success-rate adaptation ----
        sigma = 2.0
        WINDOW = 10
        success_count = 0
        stall = 0
        STALL_LIMIT = 40
        step_in_window = 0

        while eval_count < budget and _time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = _safe_call(y)
            if fy is None:
                break

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                stall = 0
            else:
                stall += 1

            step_in_window += 1
            if step_in_window >= WINDOW:
                # Adapt sigma based on success rate in this window
                rate = success_count / WINDOW
                if rate > 0.5:
                    sigma *= 1.15  # too many successes: increase step
                elif rate < 0.3:
                    sigma *= 0.85  # too few successes: decrease step
                # else keep sigma
                success_count = 0
                step_in_window = 0

            if stall >= STALL_LIMIT:
                # Stall restart: jump from current best
                if eval_count < budget and _time_ok():
                    x_new = np.clip(x + 0.5 * sigma * rng.standard_normal(dim), -5.0, 5.0)
                    fx_new = _safe_call(x_new)
                    if fx_new is not None:
                        x, fx = x_new, fx_new
                sigma *= 0.5
                stall = 0
                success_count = 0
                step_in_window = 0

        # ---- Phase 2: coordinate-descent polish ----
        step = max(sigma, 0.1)
        for _round in range(4):
            if not _time_ok() or eval_count >= budget:
                break
            for d in range(dim):
                if eval_count >= budget or not _time_ok():
                    break
                improved = False
                for sign in (1.0, -1.0):
                    if eval_count >= budget or not _time_ok():
                        break
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                    ftrial = _safe_call(trial)
                    if ftrial is None:
                        break
                    if ftrial < fx:
                        x, fx = trial, ftrial
                        improved = True
                if not improved:
                    step = max(step * 0.5, 1e-4)
            step = max(step, 5e-3)

        # ---- Phase 3: fine-grained local search (Nelder-Mead lite) ----
        # A few rounds of simplex-like refinement around the best point
        if eval_count < budget and _time_ok():
            # Simplex initialization: x and x + eps*e_i
            eps = max(sigma * 0.1, 1e-2)
            simplex = [x.copy()]
            for i in range(dim):
                vi = x.copy()
                vi[i] += eps
                simplex.append(vi)

            # Evaluate simplex vertices (x already known)
            simplex_f = [fx]
            for vi in simplex[1:]:
                fi = _safe_call(vi)
                if fi is None:
                    simplex_f = None
                    break
                simplex_f.append(fi)

            if simplex_f is not None:
                # A few Nelder-Mead iterations
                for _nm_iter in range(15):
                    if eval_count >= budget or not _time_ok():
                        break
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    x_worst = simplex[-1]
                    f_worst = simplex_f[-1]
                    x_best = simplex[0]
                    f_best = simplex_f[0]

                    # Centroid of all but worst
                    centroid = np.mean([v for v in simplex[:-1]], axis=0)

                    # Reflection
                    alpha = 1.0
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5.0, 5.0)
                    f_r = _safe_call(x_r)
                    if f_r is None:
                        break

                    if f_r < f_best:
                        # Expansion
                        gamma = 2.0
                        x_e = centroid + gamma * (centroid - x_worst)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = _safe_call(x_e)
                        if f_e is None:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        rho = 0.5
                        x_c = centroid + rho * (x_worst - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = _safe_call(x_c)
                        if f_c is None:
                            break
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                fi = _safe_call(simplex[i])
                                if fi is None:
                                    break
                                simplex_f[i] = fi
                            else:
                                continue
                            break

                    # Update best
                    if f_r < fx or (f_e is not None and f_e < fx):
                        pass  # fx updated below

                # Update global best from simplex
                best_si = int(np.argmin(simplex_f))
                if simplex_f[best_si] < fx:
                    x, fx = simplex[best_si].copy(), simplex_f[best_si]

        return x, fx
# EVOLVE-BLOCK-END
