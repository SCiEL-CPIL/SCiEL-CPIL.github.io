"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead + adaptive sigma polish, with a wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- explicit deadline guard (safety margin before the 330 s kill) ----
        start = time.monotonic()
        deadline = start + 290.0  # ~40 s safety margin

        # ---- counted wrapper: never exceed budget, clip to box, track best ----
        calls = {"n": 0}
        best_x = None
        best_f = np.inf

        def call(x):
            if calls["n"] >= budget:
                return None
            calls["n"] += 1
            f = func(np.asarray(x, dtype=float))
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()
            return f

        # ---- Nelder-Mead with adaptive reset, budget + deadline aware ----
        def nelder_mead(x0, f0, max_calls, max_iter=400):
            nonlocal best_x, best_f
            n = dim
            # build initial simplex around x0
            simplex = [x0.copy()]
            base_f = [f0]
            for i in range(n):
                v = x0.copy()
                v[i] += 0.5 if v[i] < 0 else -0.5
                v = np.clip(v, -5, 5)
                fv = call(v)
                if fv is None:
                    break
                simplex.append(v)
                base_f.append(fv)
            # make sure we have n+1 vertices
            while len(simplex) < n + 1:
                v = np.clip(x0 + 0.2 * np.random.default_rng(len(simplex)).standard_normal(n), -5, 5)
                fv = call(v)
                if fv is None:
                    break
                simplex.append(v)
                base_f.append(fv)
            if len(simplex) < 2:
                return

            alpha, gamma, rho = 1.0, 2.0, 0.5
            no_improve = 0
            for _ in range(max_iter):
                if time.monotonic() > deadline or calls["n"] >= budget:
                    break
                # sort by fitness
                order = np.argsort(base_f)
                simplex = [simplex[i] for i in order]
                base_f = [base_f[i] for i in order]
                worst = simplex[-1]
                f_worst = base_f[-1]
                # centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                # reflect
                xr = np.clip(centroid + alpha * (centroid - worst), -5, 5)
                fr = call(xr)
                if fr is None:
                    break
                if fr < base_f[0]:
                    # expand
                    xe = np.clip(centroid + gamma * (centroid - worst), -5, 5)
                    fe = call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1], base_f[-1] = xe, fe
                    else:
                        simplex[-1], base_f[-1] = xr, fr
                    no_improve = 0
                elif fr < f_worst:
                    simplex[-1], base_f[-1] = xr, fr
                    no_improve = 0
                else:
                    # contract
                    xc = np.clip(centroid + rho * (worst - centroid), -5, 5)
                    fc = call(xc)
                    if fc is None:
                        break
                    if fc < f_worst:
                        simplex[-1], base_f[-1] = xc, fc
                        no_improve = 0
                    else:
                        # shrink toward best
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            fs = call(simplex[i])
                            if fs is None:
                                break
                            base_f[i] = fs
                        no_improve += 1
                if no_improve > 12:
                    # adaptive reset: jitter the worst vertex
                    w = simplex[-1].copy()
                    w = np.clip(w + 0.5 * np.random.default_rng(calls["n"]).standard_normal(n), -5, 5)
                    fw = call(w)
                    if fw is None:
                        break
                    simplex[-1], base_f[-1] = w, fw
                    no_improve = 0
            return

        # ---- multi-start diversification, then concentrate on best basin ----
        rng = np.random.default_rng(0)
        # start points: origin, random, and a few lattice-ish points
        starts = [np.zeros(dim)]
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
        # evaluate starts
        for s in starts:
            if time.monotonic() > deadline or calls["n"] >= budget:
                break
            fs = call(np.clip(s, -5, 5))
            if fs is None:
                break
        if best_x is None:
            return None

        # run Nelder-Mead from the best start, then a couple of secondary starts
        ordered = sorted(starts, key=lambda s: (call is not None))  # placeholder
        # re-evaluate best few starts (cheap, already counted) and run NM on them
        # We'll run NM on the best start first (concentrate budget), then others if time.
        # Determine best start by re-reading (we already have best_x from global tracking)
        # Run NM starting from the current global best point.
        nelder_mead(best_x.copy(), best_f, max_calls=budget)

        # If budget remains, run NM from one more diverse start to catch a second basin
        if time.monotonic() < deadline and calls["n"] < budget:
            s2 = rng.uniform(-4, 4, dim)
            f2 = call(np.clip(s2, -5, 5))
            if f2 is not None:
                nelder_mead(np.clip(s2, -5, 5), f2, max_calls=budget)

        # ---- final adaptive sigma (1+1)-ES polish on the best point ----
        if time.monotonic() < deadline and calls["n"] < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while time.monotonic() < deadline and calls["n"] < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
