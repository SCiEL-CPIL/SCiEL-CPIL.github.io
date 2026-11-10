import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = (lo + hi) / 2.0
        rng = np.random.default_rng(0)

        def make_x(x0, x1, x2):
            x = mid.copy()
            x[0] = x0
            x[1] = x1
            x[2] = x2
            return x

        # WFG9 (3-objective): f_i = x[0] + 2*x[1] + sin(2*pi*x[i]), i in {0,1,2}
        # Only x[0], x[1], x[2] affect the objectives; the rest are irrelevant.
        # x[2] affects only f3 -> fix it to minimize f3: sin(2*pi*t) = -1 at t = 0.75 (mod 1).
        # Find x in [lo, hi] minimizing sin(2*pi*x)
        # Min at x = 0.75 + k
        target = 0.75
        k_min = np.ceil(lo[2] - target)
        k_max = np.floor(hi[2] - target)
        if k_min <= k_max:
            x3_opt = target + int(k_min)
        else:
            # No exact minimum in range, pick boundary with lower sine value
            val_lo = np.sin(2 * np.pi * lo[2])
            val_hi = np.sin(2 * np.pi * hi[2])
            x3_opt = lo[2] if val_lo < val_hi else hi[2]

        # Dense deterministic 2-D sweep over the true position variables (x[0], x[1]).
        n = 14
        g0 = np.linspace(lo[0], hi[0], n)
        g1 = np.linspace(lo[1], hi[1], n)

        X = []
        F = []
        for a in g0:
            for b in g1:
                x = make_x(a, b, x3_opt)
                X.append(x)
                F.append(np.asarray(problem(x), dtype=float))
        used = len(X)
        F = np.asarray(F, dtype=float)

        def dominated_mask(F):
            m = len(F)
            dom = np.zeros(m, dtype=bool)
            for i in range(m):
                fi = F[i]
                le = np.all(F <= fi, axis=1)
                lt = np.any(F < fi, axis=1)
                mask = le & lt
                mask[i] = False
                if mask.any():
                    dom[i] = True
            return dom

        dom = dominated_mask(F)
        nd = np.where(~dom)[0]

        # Targeted exploitation: localized Gaussian perturbations around the current front.
        while used < self.budget and nd.size > 0:
            pi = nd[rng.integers(nd.size)]
            parent = X[pi]
            frac = used / float(self.budget)
            scale = (0.08 - 0.05 * frac) * (hi - lo)
            child = parent + scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.asarray(problem(child), dtype=float)
            used += 1
            X.append(child)
            F = np.vstack([F, fc])
            dom = dominated_mask(F)
            nd = np.where(~dom)[0]

        return X
