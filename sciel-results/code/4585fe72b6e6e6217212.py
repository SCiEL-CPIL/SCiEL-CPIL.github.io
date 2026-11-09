"""Hybrid global Nelder-Mead + coordinate line search for BBOB f14."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead simplex with coordinate-wise line-search polish.

    - Global phase: several Nelder-Mead simplices from diverse seeds (exploits
      direction-adaptive convergence that handles rotated/shifted basins).
    - Local phase: Powell-style coordinate line search from the best point.
    - Explicit wall-clock deadline guard so we always return a valid best.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 325.0  # safety margin under the 330 s kill

        # ---- counters -------------------------------------------------------
        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(x))

        def time_left():
            return (t0 + DEADLINE) - time.time()

        # ---- global phase: multi-start Nelder-Mead --------------------------
        rng = np.random.default_rng(0)
        n = self.dim

        def nelder_mead(x0, fx0, max_iter, min_step=1e-7, min_diff=1e-12):
            """Nelder-Mead returning (best_x, best_f, calls_used)."""
            x = np.array(x0, dtype=float)
            fx = float(fx0)
            # build initial simplex
            simplex = [x.copy()]
            sf = [fx]
            for i in range(n):
                v = x.copy()
                d = 0.5 if x[i] >= 0 else -0.5
                v[i] = x[i] + d
                fv = f(v)
                simplex.append(v)
                sf.append(fv)

            # Nelder-Mead constants
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if time_left() < 5.0:
                    break
                order = np.argsort(sf)
                simplex = [simplex[i] for i in order]
                sf = [sf[i] for i in order]
                worst = simplex[-1]
                f_worst = sf[-1]
                best = simplex[0]
                f_best = sf[0]

                if abs(f_worst - f_best) < min_diff and np.max(np.abs(x - best)) < min_step:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + alpha * (centroid - worst)
                fr = f(xr)
                if fr < f_best:
                    # expansion
                    xe = centroid + gamma * (centroid - worst)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        sf[-1] = fe
                    else:
                        simplex[-1] = xr
                        sf[-1] = fr
                elif fr < sf[-2]:
                    simplex[-1] = xr
                    sf[-1] = fr
                else:
                    # contraction
                    xc = centroid + rho * (worst - centroid)
                    fc = f(xc)
                    if fc < f_worst:
                        simplex[-1] = xc
                        sf[-1] = fc
                    else:
                        # shrink
                        for i in range(1, n + 1):
                            simplex[i] = best + sigma * (simplex[i] - best)
                            sf[i] = f(simplex[i])
                        # keep best
                        simplex[0] = best
                        sf[0] = f_best

                # update running best
                mi = int(np.argmin(sf))
                if sf[mi] < fx:
                    fx = sf[mi]
                    x = simplex[mi].copy()

            # ensure best
            mi = int(np.argmin(sf))
            if sf[mi] < fx:
                fx = sf[mi]
                x = simplex[mi].copy()
            return x, fx

        # seeds: origin + a few diverse points
        seeds = [np.zeros(n)]
        for _ in range(4):
            seeds.append(rng.uniform(-4.5, 4.5, n))

        # allocate global budget per start (leave room for polish)
        global_budget = int(self.budget * 0.55)
        per_start = max(20, global_budget // len(seeds))

        best_x = None
        best_f = np.inf
        for s in seeds:
            if calls[0] >= self.budget - 30 or time_left() < 10.0:
                break
            fs = f(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
            # run NM from this seed
            xs, fs2 = nelder_mead(s, fs, per_start)
            if fs2 < best_f:
                best_f = fs2
                best_x = xs.copy()

        # ---- local phase: coordinate line search (Powell) -------------------
        if best_x is None:
            best_x = np.zeros(n)
            best_f = f(best_x)

        def line_search(x, d, f_start, max_iter=30, tol=1e-10):
            """1-D golden-section search along x + t*d for t in [-2, 2]."""
            a, b = -2.0, 2.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d_ = a + gr * (b - a)
            fc = f(x + c * d)
            fd = f(x + d_ * d)
            for _ in range(max_iter):
                if time_left() < 5.0:
                    break
                if fc < fd:
                    b = d_
                    d_ = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(x + c * d)
                else:
                    a = c
                    c = d_
                    fc = fd
                    d_ = a + gr * (b - a)
                    fd = f(x + d_ * d)
                if abs(b - a) < tol:
                    break
            # pick better of the two interior points
            if fc < fd:
                t = c
            else:
                t = d_
            return x + t * d

        # coordinate-wise refinement
        x = best_x.copy()
        fx = best_f
        improved = True
        while improved and calls[0] < self.budget - 10 and time_left() > 15.0:
            improved = False
            for i in range(n):
                if calls[0] >= self.budget - 10 or time_left() < 15.0:
                    break
                d = np.zeros(n)
                d[i] = 1.0
                xn = line_search(x, d, fx)
                fxn = f(xn)
                if fxn < fx - 1e-12:
                    x = xn
                    fx = fxn
                    improved = True
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # ---- final safety: ensure we used a reasonable number of calls -------
        # If we somehow used very few calls, do a light random restart polish.
        while calls[0] < self.budget - 5 and time_left() > 20.0:
            cand = best_x + 0.1 * rng.standard_normal(n)
            cand = np.clip(cand, -5, 5)
            fc = f(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
                # quick coordinate polish
                for _ in range(3):
                    for i in range(n):
                        if calls[0] >= self.budget - 5 or time_left() < 20.0:
                            break
                        d = np.zeros(n)
                        d[i] = 1.0
                        xn = line_search(best_x, d, best_f)
                        fxn = f(xn)
                        if fxn < best_f - 1e-12:
                            best_f = fxn
                            best_x = xn.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
