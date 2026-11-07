"""Multi-objective optimizer for RE34 using hypervolume-driven local search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-gap-driven L-BFGS-B search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1705.0, 12.0, 0.3])

        X_archive = []
        F_archive = []
        eval_count = 0

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10] * self.n_obj
            f = problem(x)
            eval_count += 1
            X_archive.append(np.array(x, dtype=float))
            F_archive.append(np.array(f, dtype=float))
            return f

        def is_better_than_ref(f):
            return all(fi < ref[i] for i, fi in enumerate(f))

        def get_nd_indices(F_arr):
            """Return indices of non-dominated points in F_arr (minimization)."""
            n = len(F_arr)
            if n == 0:
                return []
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0].tolist()

        def compute_hv_contribution(f, F_arr, ref_pt):
            """Compute the hypervolume contribution of point f given other points."""
            # Simplified: just check if f is better than ref and non-dominated
            if not is_better_than_ref(f):
                return 0.0
            # A rough estimate: volume of the box from f to ref
            vol = 1.0
            for i in range(self.n_obj):
                if f[i] < ref_pt[i]:
                    vol *= (ref_pt[i] - f[i])
                else:
                    return 0.0
            return vol

        # Phase 1: Stratified initial sampling
        n_init = 30
        for _ in range(n_init):
            if eval_count >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        # Phase 2: Hypervolume-gap-driven local search
        # We iteratively find the largest gap in the current front and try to fill it
        n_local_rounds = 10

        for round_idx in range(n_local_rounds):
            if eval_count >= self.budget - 5 or time.time() > deadline:
                break

            # Get current non-dominated set
            F_arr = np.array(F_archive)
            nd_idx = get_nd_indices(F_arr)

            if len(nd_idx) < 2:
                # Not enough points, do random sampling
                x = lo + (hi - lo) * np.random.rand(self.dim)
                safe_problem(x)
                continue

            # Generate candidate weight vectors targeting gaps
            # Use points on the convex hull of the non-dominated set
            F_nd = F_arr[nd_idx]
            n_nd = len(F_nd)

            # Create weight vectors pointing to gaps between adjacent ND points
            # Sort by first objective for ordering
            order = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[order]

            candidate_weights = []
            for k in range(len(F_sorted) - 1):
                # Midpoint between adjacent points
                mid = (F_sorted[k] + F_sorted[k + 1]) / 2.0
                # Weight vector pointing toward this gap
                # Use the negative of the direction from the current best to the gap
                w = np.ones(self.n_obj) / self.n_obj
                # Emphasize the direction of the gap
                for i in range(self.n_obj):
                    w[i] = 1.0 / (1.0 + abs(mid[i] - F_sorted[k][i]) + 1e-6)
                w = w / np.sum(w)
                candidate_weights.append(w)

            # Also add some random weight vectors for diversity
            for _ in range(3):
                w = np.random.dirichlet(np.ones(self.n_obj))
                candidate_weights.append(w)

            # Shuffle and process
            np.random.shuffle(candidate_weights)

            for w in candidate_weights:
                if eval_count >= self.budget - 3 or time.time() > deadline:
                    break

                # Find best starting point for this weight direction
                F_arr = np.array(F_archive)
                F_scaled = F_arr / ref
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()

                # Define scalarized objective
                def scalar_obj(x, w=w):
                    f = safe_problem(x)
                    return np.dot(w, f / ref)

                # L-BFGS-B with limited iterations
                remaining = self.budget - eval_count - 3
                if remaining < 5:
                    break

                max_iter = min(10, max(3, remaining // 3))

                try:
                    res = minimize(
                        scalar_obj, x0, method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'maxls': 15,
                                 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                except Exception:
                    pass

        # Phase 3: Final random fill if budget remains
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
