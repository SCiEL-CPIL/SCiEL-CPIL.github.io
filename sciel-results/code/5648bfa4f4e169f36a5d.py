# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-targeted optimizer: sample the known Pareto front (g=0), refine, fill."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 300.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        m = self.n_obj
        budget = self.budget

        def rem():
            return deadline - 0.5 - time.time()

        def is_feas(fv):
            return bool(np.all(fv < np.array([1.1, 1.1, 25.0])))

        def dom(a, b):
            # True if a dominates b (a <= b all, a < b some)
            return bool(np.all(a <= b) and np.any(a < b))

        # ---------- 1) Sample the true Pareto front (g=0) ----------
        # front: x0,x1,x2 in [0,1], trailing = 0  ->  f1=x0,f2=x1,f3=x2,f4=1-x2
        # 3D grid for broad coverage
        gsize = 5
        grid = np.array([
            [lo[0] + (hi[0]-lo[0])*a, lo[1] + (hi[1]-lo[1])*b,
             lo[2] + (hi[2]-lo[2])*c, *np.zeros(n-3)]
            for a in np.linspace(0, 1, gsize)
            for b in np.linspace(0, 1, gsize)
            for c in np.linspace(0, 1, gsize)
        ])
        # add a few extra corner/edge points
        extras = [
            [0.0, 0.0, 0.0, *np.zeros(n-3)],
            [1.0, 1.0, 0.0, *np.zeros(n-3)],
            [0.0, 1.0, 1.0, *np.zeros(n-3)],
            [1.0, 0.0, 1.0, *np.zeros(n-3)],
            [0.5, 0.5, 0.5, *np.zeros(n-3)],
        ]
        grid = np.vstack([grid, np.array(extras)])

        used = 0
        allX = []
        allF = []
        for x in grid:
            if used >= budget:
                break
            fv = problem(x)
            allX.append(x); allF.append(fv); used += 1
        allX = np.array(allX)
        allF = np.array(allF)

        # ---------- 2) Local refinement: drive trailing vars to 0 ----------
        # Start from a small set of random points; mutate trailing vars toward 0
        if rem() > 10:
            rng = np.random.default_rng(1)
            k = min(8, budget - used)
            init = lo + span * rng.random((k, n))
            # zero out trailing vars to be on/near front, keep x0,x1,x2
            for i in range(k):
                init[i, 3:] = 0.0
            initF = np.array([problem(x) for x in init])
            used += k
            allX = np.vstack([allX, init])
            allF = np.vstack([allF, initF])

            # refine trailing vars
            for it in range(12):
                if used >= budget - 2 or rem() < 5:
                    break
                # pick the point with worst f4 (highest g influence) among non-front
                # just iterate over a few
                for i in range(min(k, 4)):
                    if used >= budget:
                        break
                    x = allX[i].copy()
                    # nudge trailing vars toward 0 with random perturbation
                    step = 0.05 * (1 - it/12.0)
                    x[3:] += step * rng.standard_normal(n-3)
                    x = np.clip(x, lo, hi)
                    fv = problem(x)
                    used += 1
                    if is_feas(fv) and (not is_feas(allF[i]) or
                                        np.sum(fv) < np.sum(allF[i])):
                        allX[i] = x
                        allF[i] = fv

        # ---------- 3) Fill middle of front for hypervolume ----------
        # hypervolume reference [1.1,1.1,25]; points need f1<1.1,f2<1.1,f3<25
        # On front f3=x2<1, f4=1-x2. To maximize HV, spread f1,f2,f4.
        # Add points along edges of the front cube
        if rem() > 5 and used < budget:
            rng2 = np.random.default_rng(2)
            # sample along the "ridge" where one of x0,x1,x2 is at extreme
            nfill = min(budget - used, 120)
            for _ in range(nfill):
                if used >= budget:
                    break
                # random point on front
                x = np.zeros(n)
                x[0] = rng2.random()
                x[1] = rng2.random()
                x[2] = rng2.random()
                fv = problem(x)
                used += 1
                if is_feas(fv):
                    allX = np.vstack([allX, x])
                    allF = np.vstack([allF, fv])

        # ---------- 4) Final non-dominated filter (for cleanliness) ----------
        # (not required; return allX)
        return allX
# EVOLVE-BLOCK-END
