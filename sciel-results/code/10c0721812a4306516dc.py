# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Space-filling Latin Hypercube Sampling (LHS) for initial population.
      2. Greedy local refinement (Nelder-Mead on scalarized objectives) on
         non-dominated points to tighten the front.
      3. Deadline guard to stay well within the 300s per-evaluation limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objectives
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            return x, f

        def is_non_dominated(F):
            """Return boolean mask of non-dominated rows in F (n x n_obj)."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                # j dominates i if F[j] <= F[i] and F[j] < F[i] in at least one
                dominated = np.any(
                    np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1),
                )
                # Exclude self-comparison: F[i] <= F[i] and F[i] < F[i] is false,
                # so self never dominates. But we need to exclude i from the check.
                # Actually the above includes i: F[i] <= F[i] is True, F[i] < F[i] is False,
                # so any(...) is False for self. Good.
                if dominated:
                    mask[i] = False
            return mask

        def fast_non_dominated(F):
            """Faster non-dominated sorting for small arrays."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            # Vectorized: for each i, check if exists j != s.t. F[j] <= F[i] & any F[j] < F[i]
            # Use broadcasting for n <= ~500
            if n <= 600:
                # F[j] <= F[i] for all objectives
                le = np.all(F[:, None, :] <= F[None, :, :], axis=2)  # n x n
                lt = np.any(F[:, None, :] < F[None, :, :], axis=2)   # n x n
                # j dominates i if le[j,i] and lt[j,i]
                dom = le.T & lt.T  # dom[i] is True if some j dominates i
                # Exclude self
                np.fill_diagonal(dom, False)
                mask = ~dom
            else:
                mask = is_non_dominated(F)
            return mask

        # Phase 1: Latin Hypercube Sampling
        n_lhs = min(budget, 80)
        if n_lhs < 10:
            n_lhs = min(budget, 10)

        rng = np.random.RandomState(42)
        # LHS: for each dimension, partition [0,1] into n_lhs equal intervals,
        # pick one point per interval with a random permutation
        lxs = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lxs[:, d] = (perm + rng.random(n_lhs)) / n_lhs

        for x in lxs:
            if eval_count >= budget:
                break
            xx, ff = evaluate(x)
            X_all.append(xx)
            F_all.append(ff)

        X_all = np.array(X_all)
        F_all = np.array(F_all)

        # Phase 2: Local refinement on non-dominated points
        # Use Nelder-Mead on scalarized objectives (weighted sum with random weights)
        # to find better points near the front.

        def scalarize(x, weights):
            f = np.asarray(problem(np.clip(x, lo, hi)), dtype=float)
            return float(np.dot(weights, f))

        def nelder_mead_scalar(x0, weights, max_iter=50):
            """Simple Nelder-Mead minimizing weighted sum of objectives."""
            dim = len(x0)
            # Initial simplex
            simplex = [np.array(x0, dtype=float)]
            for i in range(dim):
                dx = np.zeros(dim)
                step = 0.05 * (hi[i] - lo[i]) if hi[i] > lo[i] else 0.05
                dx[i] = step
                simplex.append(np.clip(x0 + dx, lo, hi))

            f_vals = [scalarize(s, weights) for s in simplex]

            for _ in range(max_iter):
                if eval_count >= budget:
                    break
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = scalarize(xr, weights)
                eval_count += 1

                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    fe = scalarize(xe, weights)
                    eval_count += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = scalarize(xc, weights)
                    eval_count += 1
                    if fc < f_vals[-1]:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_vals[i] = scalarize(simplex[i], weights)
                            eval_count += 1
                            if eval_count >= budget:
                                break
                        if eval_count >= budget:
                            break

                # Record the best point in the simplex
                best_idx = np.argmin(f_vals)
                X_all.append(simplex[best_idx])
                F_all.append(np.asarray(problem(simplex[best_idx]), dtype=float))

            return simplex[np.argmin(f_vals)]

        # Get non-dominated points for refinement
        F_arr = np.array(F_all)
        X_arr = np.array(X_all)
        nd_mask = fast_non_dominated(F_arr)
        nd_indices = np.where(nd_mask)[0]

        # Select up to 10 non-dominated points for local search
        max_refine = min(10, len(nd_indices))
        if max_refine > 0:
            # Use deterministic selection: pick spread-out points
            if len(nd_indices) > max_refine:
                step = len(nd_indices) / max_refine
                selected = [nd_indices[int(i * step)] for i in range(max_refine)]
            else:
                selected = list(nd_indices)

            # Generate diverse weight vectors for scalarization
            weight_sets = []
            # Equal weights
            weight_sets.append(np.ones(n_obj) / n_obj)
            # Emphasize each objective
            for k in range(n_obj):
                w = np.ones(n_obj) * 0.1
                w[k] = 0.8
                weight_sets.append(w / np.sum(w))
            # Random weights
            for _ in range(3):
                w = rng.dirichlet(np.ones(n_obj))
                weight_sets.append(w)

            for idx in selected:
                if eval_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                x0 = X_arr[idx].copy()
                for w in weight_sets:
                    if eval_count >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    nelder_mead_scalar(x0, w, max_iter=30)

        # Ensure we don't exceed budget
        # (eval_count is tracked; if we're under budget, we could do more,
        #  but the deadline guard is the primary safety)

        # Final: return the last X (interface requires returning something)
        if len(X_all) > 0:
            return np.array(X_all[-1])
        return np.zeros(dim)
# EVOLVE-BLOCK-END
