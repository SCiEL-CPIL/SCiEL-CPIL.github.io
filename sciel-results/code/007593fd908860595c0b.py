"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-objective optimizer: global evolutionary search + local Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Use global np.random to respect the seed set by the evaluator
        # This ensures seeds 0 and 1 explore different regions.

        archive_X = []
        archive_F = []
        used = 0

        def evaluate_and_add(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1

            # Check dominance against existing archive
            if len(archive_F) == 0:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return

            F_arr = np.array(archive_F)

            # Is new point dominated by any archive point?
            # dominated if exists j: F[j] <= f and F[j] < f
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if np.any(mask_le & mask_lt):
                return  # Dominated, discard

            # Remove archive points dominated by new point
            # dominated if exists j: f <= F[j] and f < F[j]
            mask_le2 = np.all(f <= F_arr, axis=1)
            mask_lt2 = np.any(f < F_arr, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]

            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)

            archive_X.append(x.copy())
            archive_F.append(f.copy())

        # Phase 1: Initialization
        # Mix of random and boundary points
        n_init = 40
        X_init = np.zeros((n_init, dim))
        # 20 random
        X_init[:20] = lo + span * np.random.random((20, dim))
        # 20 boundary/corner points
        for i in range(20):
            # Randomly choose low or high for each dimension
            X_init[20 + i] = np.where(np.random.random(dim) < 0.5, lo, hi)

        for x in X_init:
            evaluate_and_add(x)

        # Phase 2: Global Search (Evolutionary)
        # Use remaining budget for global search, leaving some for local polish
        # Reserve 80 evaluations for local polish
        n_local_reserve = 80
        n_global = min(80, budget - used - n_local_reserve)

        if n_global > 0:
            for _ in range(n_global):
                if time.time() > deadline:
                    break
                if used >= budget - n_local_reserve:
                    break

                # Select parent from archive
                if len(archive_X) == 0:
                    parent = np.random.uniform(lo, hi)
                else:
                    # Select random parent from archive
                    idx = np.random.randint(len(archive_X))
                    parent = np.array(archive_X[idx])

                # Mutate
                # Adaptive step size: start large, shrink
                progress = used / budget
                scale = 0.5 * (1.0 - progress) + 0.05
                child = parent + scale * span * np.random.standard_normal(dim)

                evaluate_and_add(child)

        # Phase 3: Local Polish
        # Take top points from archive and refine them using local optimization
        if len(archive_X) > 0:
            # Sort archive by first objective to pick diverse starting points
            F_arr = np.array(archive_F)
            order = np.argsort(F_arr[:, 0])

            # Select 4 points: min f1, max f1, and 2 in between
            n_select = min(4, len(order))
            if n_select <= 2:
                indices_to_optimize = list(range(len(order)))
            else:
                indices_to_optimize = [
                    order[0],
                    order[len(order) // 3],
                    order[2 * len(order) // 3],
                    order[-1]
                ]

            # Ensure uniqueness
            seen = set()
            unique_indices = []
            for idx in indices_to_optimize:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            indices_to_optimize = unique_indices

            for idx in indices_to_optimize:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])

                # Define local objective: Weighted Sum
                # Use weights inverse to f0 to target the region around x0
                w = np.array([1.0 / f0[0] if f0[0] > 1e-10 else 1.0,
                              1.0 / f0[1] if f0[1] > 1e-10 else 1.0])
                w = w / np.sum(w)

                def obj(x):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    return np.dot(w, f)

                # Run Nelder-Mead
                # We have limited budget left, so restrict maxfev
                remaining = budget - used
                maxfev = min(20, remaining)
                if maxfev < 2:
                    continue

                try:
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': maxfev // 2, 'maxfev': maxfev,
                                            'xatol': 1e-4, 'fatol': 1e-4,
                                            'adaptive': True})
                    # The result x is already clipped in obj, but let's be safe
                    x_best = np.clip(res.x, lo, hi)
                    evaluate_and_add(x_best)
                except Exception:
                    pass

        # Return the archive points (the return value is ignored by evaluator,
        # but we must return something)
        if archive_X:
            return np.array(archive_X)
        return np.array([lo])
# EVOLVE-BLOCK-END
