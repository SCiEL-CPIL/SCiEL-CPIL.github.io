"""Multi-objective optimizer for RE34 using HV-guided search with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial structured sampling: corners + random
        X_list = []
        F_list = []

        # Add corners for better initial coverage
        for d in range(self.dim):
            for val in [lo[d], hi[d]]:
                x = lo.copy()
                x[d] = val
                f = np.array(problem(x))
                X_list.append(x)
                F_list.append(f)

        # Add random points to fill up initial population
        n_init = min(40, self.budget // 5)
        n_corners = 2 * self.dim
        n_random = max(0, n_init - n_corners)
        for i in range(n_random):
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            X_list.append(x)
            F_list.append(f)

        used = len(X_list)
        step_size = 0.25

        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute approximate HV contribution for each point
            valid = np.all(F_arr < ref, axis=1)
            if np.sum(valid) == 0:
                parent_idx = rng.integers(n)
            else:
                # For valid points, compute contribution
                Fv = F_arr[valid]
                nv = len(Fv)
                contrib = np.zeros(n)
                if nv == 1:
                    contrib[valid] = np.prod(np.maximum(0, ref - Fv[0]))
                else:
                    # Approximate: box volume * uniqueness weight
                    for i in range(nv):
                        box_vol = np.prod(np.maximum(0, ref - Fv[i]))
                        # Uniqueness: distance to nearest neighbor
                        dists = np.linalg.norm(Fv - Fv[i], axis=1)
                        dists[i] = np.inf
                        min_dist = np.min(dists) if nv > 1 else 1.0
                        scale = np.max(np.linalg.norm(Fv - Fv[0], axis=1)) + 1e-10
                        weight = 1.0 / (1.0 + 0.5 * min_dist / scale)
                        contrib[valid][i] = box_vol * weight

                if np.max(contrib) > 0:
                    top_k = min(5, int(np.sum(contrib > 0)))
                    if top_k > 0:
                        valid_idx = np.where(contrib > 0)[0]
                        top_idx = valid_idx[np.argsort(contrib[valid_idx])][-top_k:]
                        w = contrib[top_idx]
                        w = w / w.sum()
                        parent_idx = rng.choice(top_idx, p=w)
                    else:
                        parent_idx = rng.integers(n)
                else:
                    parent_idx = rng.integers(n)

            parent = X_list[parent_idx]
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.array(problem(child))
            X_list.append(child)
            F_list.append(fc)
            used += 1

            # Adapt step size
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                if not dominated:
                    step_size = max(0.05, step_size * 0.92)
                else:
                    step_size = min(1.0, step_size * 1.08)
            else:
                step_size = min(1.0, step_size * 1.15)

        return X_list
# EVOLVE-BLOCK-END
