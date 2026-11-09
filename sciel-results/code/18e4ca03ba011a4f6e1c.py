# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Global Sobol scan + coordinate descent + Nelder-Mead refinement.

    Budget is split: a cheap quasi-random global search to find the basin
    (fast, since AOCC rewards early progress), then a fast local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        n = self.budget
        t_start = time.time()
        DEADLINE = 300.0          # hard evaluator wall-clock limit
        SAFE = 5.0                # safety margin so we always return in time

        # ---- deterministic Sobol quasi-random points (cheap, no RNG calls) ----
        gen = qmc.Sobol(d=dim, scramble=False)
        xg = gen.random(n) * 10.0 - 5.0

        xbest = None
        fbest = np.inf
        for i in range(n):
            xi = xg[i]
            fi = func(xi)
            if fi < fbest:
                fbest = fi
                xbest = xi
            if i % 25 == 24 and (time.time() - t_start) > DEADLINE - SAFE:
                break

        # ---- local refinement (fast; early progress is already banked) ----
        if xbest is not None:
            xbest, fbest = self._refine(func, xbest, fbest, dim, t_start, DEADLINE, SAFE)

        return xbest, fbest

    def _refine(self, func, x0, f0, dim, t_start, DEADLINE, SAFE):
        x = np.array(x0, dtype=float)
        fx = f0

        def timed_out():
            return (time.time() - t_start) > DEADLINE - SAFE

        # coordinate descent: cheap, handles the sharp-ridge anisotropy well
        for _ in range(40):
            if timed_out():
                break
            improved = False
            for j in range(dim):
                for delta in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                    for s in (1.0, -1.0):
                        cand = x.copy()
                        cand[j] = np.clip(cand[j] + s * delta, -5.0, 5.0)
                        fc = func(cand)
                        if fc < fx - 1e-12:
                            x, fx = cand, fc
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        # Nelder-Mead: converge to the sharp minimum
        simplex = [x.copy()]
        for j in range(dim):
            v = x.copy()
            step = 0.25 if abs(x[j]) < 4.75 else 0.2
            v[j] = np.clip(v[j] + step, -5.0, 5.0)
            simplex.append(v)
        fvals = [func(v) for v in simplex]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        for _ in range(300):
            if timed_out():
                break
            order = np.argsort(fvals)
            simplex = [simplex[k] for k in order]
            fvals = [fvals[k] for k in order]
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            fw = fvals[-1]
            x_new = centroid + alpha * (centroid - worst)
            x_new = np.clip(x_new, -5.0, 5.0)
            fn = func(x_new)
            if fn < fvals[0]:
                x_new = centroid + gamma * (centroid - worst)
                x_new = np.clip(x_new, -5.0, 5.0)
                fg = func(x_new)
                if fg < fn:
                    simplex[-1], fvals[-1] = x_new, fg
                else:
                    simplex[-1], fvals[-1] = x_new, fn
            elif fn < fw:
                simplex[-1], fvals[-1] = x_new, fn
            else:
                x_new = centroid + rho * (worst - centroid)
                x_new = np.clip(x_new, -5.0, 5.0)
                fr = func(x_new)
                if fr < fw:
                    simplex[-1], fvals[-1] = x_new, fr
                else:
                    sigma = 2.0
                    for k in range(1, len(simplex)):
                        nk = 0.5 * (simplex[k] + simplex[0])
                        nk = np.clip(nk, -5.0, 5.0)
                        fnk = func(nk)
                        simplex[k], fvals[k] = nk, fnk
                        if timed_out():
                            break
                    if timed_out():
                        break

            best = int(np.argmin(fvals))
            if fvals[best] < fx:
                fx, x = fvals[best], simplex[best].copy()

        return x, fx
# EVOLVE-BLOCK-END
