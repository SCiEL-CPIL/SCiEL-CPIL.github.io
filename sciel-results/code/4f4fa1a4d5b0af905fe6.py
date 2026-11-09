"""Optimizer for BBOB f13 (Sharp ridge): Nelder-Mead + Powell conjugate-direction polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NM simplex (ridge-adaptive) + Powell conjugate-direction refinement + restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # 300s limit, 15s safety margin
        calls = 0
        dim = self.dim
        n = dim

        def call(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        def time_ok():
            return time.time() < deadline - 2.0

        def budget_ok():
            return calls < self.budget

        def alive():
            return budget_ok() and time_ok()

        # ------------------------------------------------------------------
        # Powell conjugate-direction line search (exploits flat ridge dirs)
        # ------------------------------------------------------------------
        def line_search(x, d, fx):
            """Golden-section + parabolic 1-D minimization along d (box-clipped)."""
            if calls >= self.budget or not time_ok():
                return x, fx
            best_x = x.copy()
            best_f = fx
            best_t = 0.0
            # Initial bracket: find t_lo (0) and t_hi with f(t_hi) < f(0)
            t_hi = 1.0
            x_hi = np.clip(x + t_hi * d, -5, 5)
            if np.array_equal(x_hi, x):
                return x, fx
            f_hi = call(x_hi)
            if f_hi >= fx:
                # function increases; try a smaller step to bracket a dip
                t_hi = 0.5
                x_hi = np.clip(x + t_hi * d, -5, 5)
                if np.array_equal(x_hi, x):
                    return x, fx
                f_hi = call(x_hi)
                if f_hi >= fx:
                    return x, fx
            t_lo, f_lo = 0.0, fx
            # Expand until f is increasing (bracket the minimum)
            while alive() and f_hi < f_lo:
                t_lo, f_lo = t_hi, f_hi
                t_hi *= 2.0
                x_hi = np.clip(x + t_hi * d, -5, 5)
                if np.array_equal(x_hi, x):
                    break
                f_hi = call(x_hi)
                if f_hi >= f_lo:
                    break
            # Now bracket [t_lo, t_hi] contains a minimum (f_lo <= f_hi).
            # Golden-section refinement.
            if f_lo > f_hi:
                t_lo, t_hi = t_hi, t_lo
                f_lo, f_hi = f_hi, f_lo
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = t_lo, t_hi
            # ensure a < b
            if a > b:
                a, b = b, a
            c = b - invphi * (b - a)
            dpt = a + invphi * (b - a)
            if not alive():
                return best_x, best_f
            x_c = np.clip(x + c * d, -5, 5)
            f_c = call(x_c)
            if not alive():
                return best_x, best_f
            x_d = np.clip(x + dpt * d, -5, 5)
            f_d = call(x_d)
            iters = 0
            while alive() and (b - a) > 1e-7 * max(1.0, abs(a), abs(b)) and iters < 40:
                iters += 1
                if f_c <= f_d:
                    b = dpt
                    dpt = c
                    f_d = f_c
                    c = b - invphi * (b - a)
                    x_c = np.clip(x + c * d, -5, 5)
                    f_c = call(x_c)
                else:
                    a = c
                    c = dpt
                    f_c = f_d
                    dpt = a + invphi * (b - a)
                    x_d = np.clip(x + dpt * d, -5, 5)
                    f_d = call(x_d)
                # Track the best seen
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                    best_t = c
                if f_d < best_f:
                    best_f = f_d
                    best_x = x_d.copy()
                    best_t = dpt
            return best_x, best_f

        def powell(x, fx, max_cycles=4):
            """Powell method: build conjugate directions, slide along ridge."""
            directions = np.eye(dim)
            for cyc in range(max_cycles):
                if not alive():
                    break
                start = x.copy()
                start_f = fx
                for i in range(dim):
                    if not alive():
                        break
                    x, fx = line_search(x, directions[i], fx)
                # New conjugate direction from net movement this cycle
                new_dir = x - start
                if np.linalg.norm(new_dir) > 1e-12:
                    new_dir = new_dir / np.linalg.norm(new_dir)
                    # Replace the direction that gave the largest improvement
                    # (approx: replace last direction with the conjugate one)
                    directions[-1] = new_dir
                else:
                    break
                if not alive():
                    break
                x, fx = line_search(x, directions[-1], fx)
                if fx >= start_f - 1e-12:
                    break
            return x, fx

        # ------------------------------------------------------------------
        # Nelder-Mead simplex
        # ------------------------------------------------------------------
        def build_simplex(x):
            simplex = np.zeros((n + 1, n))
            simplex[0] = x
            for i in range(n):
                simplex[i + 1] = x.copy()
                simplex[i + 1][i] += 0.5 if x[i] < 4.5 else -0.5
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
            return simplex

        def nm_iterate(simplex, fvals, best_f, best_x):
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            centroid = np.mean(simplex[:-1], axis=0)

            if not alive():
                return simplex, fvals, best_f, best_x, 0
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = call(xr)
            if not alive():
                return simplex, fvals, best_f, best_x, 0

            accepted = 0
            if fr < fvals[0]:
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                if alive():
                    fe = call(xe)
                else:
                    fe = fr
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                accepted = 1
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
                accepted = 1
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                if alive():
                    fc = call(xc)
                else:
                    fc = fvals[-1]
                if fc < min(fr, fvals[-1]):
                    simplex[-1], fvals[-1] = xc, fc
                    accepted = 1
                else:
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        if alive():
                            fvals[i] = call(simplex[i])
                        else:
                            break
                    accepted = 0

            idx = np.argmin(fvals)
            if fvals[idx] < best_f:
                best_f = fvals[idx]
                best_x = simplex[idx].copy()
            return simplex, fvals, best_f, best_x, accepted

        # ------------------------------------------------------------------
        # Main loop: basin find -> NM -> Powell polish -> occasional restart
        # ------------------------------------------------------------------
        rng = np.random.default_rng(0)

        # Phase 1: small random basin finder (early progress)
        best_x = None
        best_f = np.inf
        n_init = 12
        for _ in range(n_init):
            if not alive():
                break
            x = rng.uniform(-5, 5, dim)
            fx = call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call(best_x)

        # Phase 2: main optimization loop with restarts
        stall = 0
        max_stall = 25
        restarts = 0
        max_restarts = 6

        while alive() and restarts <= max_restarts:
            # Build simplex around current best and run NM for a while
            simplex = build_simplex(best_x)
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if not alive():
                    break
                fvals[i] = call(simplex[i])
            idx = np.argmin(fvals)
            if fvals[idx] < best_f:
                best_f = fvals[idx]
                best_x = simplex[idx].copy()

            local_stall = 0
            while alive() and local_stall < max_stall:
                simplex, fvals, best_f, best_x, accepted = nm_iterate(
                    simplex, fvals, best_f, best_x)
                local_stall = local_stall + 1 if accepted == 0 else 0
                # Periodically polish with Powell to exploit the ridge
                if local_stall in (8, 18) or local_stall >= max_stall:
                    px, pf = powell(best_x, best_f, max_cycles=3)
                    if pf < best_f:
                        best_f = pf
                        best_x = px
                        # Rebuild simplex around the improved point
                        simplex = build_simplex(best_x)
                        for i in range(n + 1):
                            if not alive():
                                break
                            fvals[i] = call(simplex[i])
                        idx = np.argmin(fvals)
                        if fvals[idx] < best_f:
                            best_f = fvals[idx]
                            best_x = simplex[idx].copy()
                        local_stall = 0

            # Final Powell polish after NM stalls
            if alive():
                px, pf = powell(best_x, best_f, max_cycles=5)
                if pf < best_f:
                    best_f = pf
                    best_x = px

            if not alive():
                break

            # Restart: perturb best point and try a new basin
            restarts += 1
            pert = rng.normal(0.0, 1.0, dim)
            cand = np.clip(best_x + pert, -5, 5)
            cf = call(cand)
            # Also try a couple of fresh random points
            for _ in range(3):
                if not alive():
                    break
                r = rng.uniform(-5, 5, dim)
                rf = call(r)
                if rf < cf:
                    cand, cf = r, rf
            if cf < best_f:
                best_f = cf
                best_x = cand.copy()

        # Final polish with remaining budget
        if alive():
            px, pf = powell(best_x, best_f, max_cycles=6)
            if pf < best_f:
                best_f = pf
                best_x = px

        return best_x, best_f
# EVOLVE-BLOCK-END
