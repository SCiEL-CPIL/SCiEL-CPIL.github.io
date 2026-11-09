"""Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with adaptive perturbation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0          # evaluator kills at 330s; keep a 30s margin
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        # ---------- helpers ----------
        def pareto_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            F_arr = np.asarray(F, dtype=float)
            keep = []
            for i in range(n):
                fi = F_arr[i]
                better = np.all(F_arr <= fi, axis=1) & np.any(F_arr < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        # ---------- main loop ----------
        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # seed points: uniform random spread
        n_init = min(10, budget)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            add(x, problem(x))

        used = len(F)
        # adaptive perturbation scale: start wide, shrink gradually
        scale = 0.30
        min_scale = 0.03

        while used < budget:
            # deadline guard
            if time.time() > deadline - 5.0:
                break

            # choose parents: prefer non-dominated
            keep = pareto_indices(F)
            if len(keep) == 0:
                keep = list(range(len(F)))

            # Pick a random non-dominated parent (uniform over front)
            pidx = keep[rng.integers(len(keep))]

            # mutation: adaptive step with occasional large jump
            if rng.random() < 0.10:
                step = scale * 2.5
            else:
                step = scale
            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X[pidx] + delta
            child = np.clip(child, lo, hi)
            fc = problem(child)
            add(child, fc)
            used += 1

            # update scale: shrink geometrically to refine
            scale = max(min_scale, scale * 0.995)

        return X
# EVOLVE-BLOCK-END
