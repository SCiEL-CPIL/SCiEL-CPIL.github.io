# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1) EA with adaptive mutation and hypervolume-based replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random

        pop_size = min(30, self.budget)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume approximation
        ref = np.array([3.0, 5.0, 7.0])

        def hv_contribution(F, idx):
            """Approximate hypervolume contribution of point idx in F."""
            n = len(F)
            # A point contributes to HV if it is non-dominated and its "box"
            # (from the point to ref, minus boxes of other non-dominated points) is positive.
            # Simple approximation: volume of box from F[idx] to ref, weighted by
            # how many points dominate it (0 if non-dominated).
            box_vol = np.prod(np.maximum(ref - F[idx], 0))
            # Penalize if dominated
            dom_count = 0
            for j in range(n):
                if j == idx:
                    continue
                if np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    dom_count += 1
            if dom_count > 0:
                return 0.0
            return box_vol

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set (first front)
            n = len(X)
            is_nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nondom[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_nondom[j] = False

            nondom_idx = np.where(is_nondom)[0]

            if len(nondom_idx) > 0:
                parent_idx = nondom_idx[rng.randint(len(nondom_idx))]
            else:
                parent_idx = rng.randint(n)

            parent = X[parent_idx]

            # Adaptive mutation: shrink sigma over time
            progress = used / self.budget
            sigma = 0.15 * (1.0 - 0.7 * progress) + 0.02
            child = parent + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Check if child dominates any existing point
            replaced = False
            for i in range(n):
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break

            if not replaced:
                # Replace the point with lowest hypervolume contribution
                # among non-dominated points, or among all if none are nondominated
                if len(nondom_idx) > 0:
                    candidates = nondom_idx
                else:
                    candidates = np.arange(n)

                best_hv = -1
                worst_idx = candidates[0]
                for c in candidates:
                    hv = hv_contribution(F, c)
                    if hv < best_hv:
                        best_hv = hv
                        worst_idx = c

                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
