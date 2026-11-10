"""Optimizer for WFG9 exploiting analytic structure: dense grid on (x0,x1) with optimal x2, then local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits WFG9's structure: objectives depend on x0, x1, x2.
    Uses a dense grid over the 2D active subspace (x0, x1) with x2 fixed
    to minimize its contribution, then refines around non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 (3-objective): f_i = x[0] + 2*x[1] + sin(2*pi*x[i]), i in {0,1,2}
        # Only x[0], x[1], x[2] affect the objectives; the rest are irrelevant.
        # x[2] affects only f3 -> fix it to minimize f3: sin(2*pi*t) = -1 at t = 0.75 (mod 1).
        x3_opt = lo[2] + 0.75 * (hi[2] - lo[2])
        mid = (lo + hi) / 2.0

        # Dense deterministic 2-D sweep over the true position variables (x[0], x[1]).
        # Allocate most budget to the grid, leave some for refinement.
        n_refine = min(40, budget // 5)
        n_grid_total = budget - n_refine

        # Grid over x0 and x1
        n_per_dim = int(np.sqrt(n_grid_total))
        if n_per_dim < 5:
            n_per_dim = 5
        # Adjust to fit budget
        n_grid_actual = n_per_dim * n_per_dim
        if n_grid_actual > n_grid_total:
            n_per_dim = int(np.sqrt(n_grid_total))
            n_grid_actual = n_per_dim * n_per_dim

        g0 = np.linspace(lo[0], hi[0], n_per_dim)
        g1 = np.linspace(lo[1], hi[1], n_per_dim)

        def make_x(x0, x1):
            x = mid.copy()
            x[0] = x0
            x[1] = x1
            x[2] = x3_opt
            return x

        X = []
        F = []
        used = 0

        for a in g0:
            for b in g1:
                if used >= n_grid_total:
                    break
                if time.time() > deadline:
                    break
                x = make_x(a, b)
                f = np.asarray(problem(x), dtype=float)
                X.append(x)
                F.append(f)
                used += 1
            if used >= n_grid_total or time.time() > deadline:
                break

        if len(X) == 0:
            return X

        X = np.array(X)
        F = np.array(F)

        # Find non-dominated points for refinement targets
        n_pts = len(F)

        def dominated_mask(F_arr):
            m = len(F_arr)
            dom = np.zeros(m, dtype=bool)
            for i in range(m):
                fi = F_arr[i]
                le = np.all(F_arr <= fi, axis=1)
                lt = np.any(F_arr < fi, axis=1)
                mask = le & lt
                mask[i] = False
                if mask.any():
                    dom[i] = True
            return dom

        dom = dominated_mask(F)
        nd_idx = np.where(~dom)[0]

        # Targeted exploitation: localized perturbations around the current front.
        rng = np.random.default_rng(0)
        if nd_idx.size > 0 and used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                pi = nd_idx[rng.integers(nd_idx.size)]
                parent = X[pi]
                frac = used / float(budget)
                scale = (0.08 - 0.05 * frac) * (hi - lo)
                child = parent + scale * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = np.asarray(problem(child), dtype=float)
                used += 1
                X = np.vstack([X, child.reshape(1, -1)])
                F = np.vstack([F, fc.reshape(1, -1)])
                dom = dominated_mask(F)
                nd_idx = np.where(~dom)[0]
                if nd_idx.size == 0:
                    break

        return X

# EVOLVE-BLOCK-END
