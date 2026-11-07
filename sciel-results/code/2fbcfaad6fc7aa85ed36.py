"""Multi-start adaptive (1+1)-ES with Nelder-Mead local exploitation for BBOB f16.

Uses Nelder-Mead simplex as the primary local search engine to exploit the
smooth structure of Weierstrass, wrapped in a global adaptive (1+1)-ES with
aggressive restarts. Nelder-Mead converges faster in the basin than random
perturbation, giving more improvement per call in the middle phase.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with Nelder-Mead local runs and polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)
        deadline = time.time() + 270.0
        n_calls = [0]

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        # ---- Nelder-Mead simplex local search -------------------------------
        def nelder_mead(start_x, max_calls=25):
            """Nelder-Mead exploiting local smoothness of Weierstrass."""
            nonlocal best_x, best_f

            simplex = [np.array(start_x, dtype=float)]
            fvals = [safe(start_x)]
            if np.isfinite(fvals[0]):
                record(simplex[0], fvals[0])

            # Build simplex with adaptive initial step
            for i in range(dim):
                pt = np.array(start_x, dtype=float)
                step = 0.25
                pt[i] += step
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)
                fvals.append(safe(pt))
                if np.isfinite(fvals[-1]):
                    record(pt, fvals[-1])
                if n_calls[0] >= budget or time.time() > deadline:
                    return

            alpha, gamma, rho, sigma_c = 1.0, 2.0, 0.5, 0.5
            max_iter = max(1, max_calls - (dim + 1))

            for _ in range(max_iter):
                if n_calls[0] >= budget or time.time() > deadline:
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                if np.isfinite(fvals[0]) and np.isfinite(fvals[-1]):
                    if fvals[-1] - fvals[0] < 1e-12:
                        break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                fr = safe(xr)

                if np.isfinite(fr) and fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = safe(xe)
                    if np.isfinite(fe) and fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif np.isfinite(fr) and fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if np.isfinite(fr) and fr < fvals[-1]:
                        xc = np.clip(centroid + sigma_c * (xr - centroid), -5.0, 5.0)
                    else:
                        xc = np.clip(centroid - sigma_c * (centroid - simplex[-1]), -5.0, 5.0)
                    fc = safe(xc)
                    if np.isfinite(fc) and fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe(simplex[i])
                            if n_calls[0] >= budget or time.time() > deadline:
                                return

                for x, f in zip(simplex, fvals):
                    if np.isfinite(f) and f < best_f:
                        record(x, f)

        # ---- Phase 1: Diverse Initial Starts --------------------------------
        starts = [np.zeros(dim)]
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        for sign in [1, -1]:
            starts.append(np.full(dim, sign * 2.0))
            starts.append(np.full(dim, sign * 4.0))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            f = safe(s)
            if np.isfinite(f):
                record(s, f)

        # ---- Phase 2: Adaptive (1+1)-ES with NM local runs ------------------
        main_limit = int(n_calls[0] + (budget - n_calls[0]) * 0.82)

        sigma = 1.5
        window = []
        win_size = 10
        stall_threshold = 25
        max_restarts = 12
        restart_count = 0
        no_improve = 0

        while n_calls[0] < main_limit and time.time() < deadline:
            no_improve = 0
            while (n_calls[0] < main_limit and time.time() < deadline
                   and no_improve <= stall_threshold):
                cand = best_x + sigma * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_x = cand.copy()
                    best_f = f
                    window.append(1)
                    no_improve = 0
                else:
                    window.append(0)
                    no_improve += 1

                if len(window) >= win_size:
                    sr = sum(window) / win_size
                    if sr > 0.35:
                        sigma *= 1.3
                    elif sr < 0.12:
                        sigma *= 0.75
                    window = window[-win_size // 2:]
                sigma = np.clip(sigma, 1e-9, 6.0)

            # On stall: exploit local structure with Nelder-Mead, then restart
            if (n_calls[0] < main_limit and time.time() < deadline
                    and no_improve > stall_threshold):
                # Local exploitation of the current basin with NM
                if n_calls[0] < main_limit and time.time() < deadline:
                    nm_budget = min(25, max(8, (main_limit - n_calls[0]) // 4))
                    nelder_mead(best_x, max_calls=nm_budget)

                restart_count += 1
                if restart_count > max_restarts:
                    break

                if restart_count % 2 == 1:
                    start_x = rng.uniform(-4.0, 4.0, dim)
                else:
                    start_x = best_x + rng.uniform(-1.2, 1.2, dim)
                start_x = np.clip(start_x, -5.0, 5.0)
                f = safe(start_x)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = start_x.copy()

                sigma = 1.0
                window = []

        # ---- Phase 3: Final Nelder-Mead polish on the best point -------------
        # Use remaining budget for a high-precision NM run near the global best.
        if n_calls[0] < budget and time.time() < deadline:
            remaining = budget - n_calls[0]
            if remaining > 10:
                nelder_mead(best_x, max_calls=min(remaining, 40))

        # ---- Phase 4: Ultra-fine random polish --------------------------------
        if n_calls[0] < budget and time.time() < deadline:
            sigma_p = 1e-4
            for _ in range(budget - n_calls[0]):
                if n_calls[0] >= budget or time.time() > deadline:
                    break
                cand = best_x + sigma_p * rng.standard_normal(dim)
                cand = np.clip(cand, -5.0, 5.0)
                f = safe(cand)
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = cand.copy()
                    sigma_p *= 0.9
                else:
                    sigma_p *= 0.95
                sigma_p = max(sigma_p, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
