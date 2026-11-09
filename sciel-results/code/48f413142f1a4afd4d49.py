"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton2d(n, base1=2, base2=3):
    """Generate n 2D low-discrepancy (Halton) points in [0,1]^2."""
    def _halton(n, base):
        out = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            r, i2 = 0.0, i
            while i2 > 0:
                f /= base
                r += f * (i2 % base)
                i2 //= base
            out[i - 1] = r
        return out
    return np.column_stack((_halton(n, base1), _halton(n, base2)))


class Optimizer:
    """Structure-exploiting multi-objective optimizer for DTLZ2.

    Places exact on-front points using a 2D Halton sequence over the angle
    parameters, then refines the non-dominated set with local search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before the 300 s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def is_deadline():
            return time.time() > deadline

        def dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def is_nd(f, F):
            return not dominated(f, F)

        # ---- Phase 1: exact on-front anchors via 2D Halton over angles ----
        n_angles = n_obj - 1  # = 2 for the 3-objective DTLZ2
        # Reserve some budget for refinement
        n_anchor = min(150, max(50, budget // 2))

        pts = _halton2d(n_anchor)  # in [0,1]^2
        # Map [0,1]^2 -> angles in [0, pi/2]
        th1 = pts[:, 0] * (np.pi / 2.0)
        th2 = pts[:, 1] * (np.pi / 2.0)

        # Build full decision vectors: angles -> x[0..n_angles-1], feasibility -> 0.5
        X = np.zeros((n_anchor, dim))
        X[:, 0] = th1
        X[:, 1] = th2
        # Feasibility variables (x[2..dim-1]) set to midpoint
        if dim > n_angles:
            X[:, n_angles:] = 0.5 * (lo[n_angles:] + hi[n_angles:])

        F = np.array([problem(x) for x in X])
        used = n_anchor

        # Keep only points strictly better than reference in every objective
        ref = np.array([2.5] * n_obj)
        valid = np.all(F < ref, axis=1)
        X = X[valid]
        F = F[valid]

        # ---- Phase 2: local refinement of non-dominated points ----
        def non_dominated_indices(F):
            if F.shape[0] == 0:
                return np.array([], dtype=int)
            mask = np.zeros(F.shape[0], dtype=bool)
            for i in range(F.shape[0]):
                if not dominated(F[i], F):
                    mask[i] = True
            return np.where(mask)[0]

        step = 0.05
        while used < budget and not is_deadline():
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([int(np.argmin(np.sum(F, axis=1)))])
            # Pick a non-dominated point to refine (cycle / random)
            idx = nd_idx[used % len(nd_idx)]
            base = X[idx]

            # Try a few random perturbations around the base point
            best_child = None
            best_fc = None
            for _ in range(3):
                if used >= budget or is_deadline():
                    break
                child = base + step * (hi - lo) * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                if np.all(fc < ref):
                    if best_child is None or np.sum(fc) < np.sum(best_fc):
                        best_child, best_fc = child, fc

            if best_child is not None and is_nd(best_fc, F):
                # Replace the worst (most dominated / largest sum) point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = best_child
                F[worst] = best_fc

        return X
# EVOLVE-BLOCK-END
