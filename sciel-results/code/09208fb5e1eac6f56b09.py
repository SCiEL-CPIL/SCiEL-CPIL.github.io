"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Tchebycheff decomposition with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

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

            if len(archive_F) == 0:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return

            F_arr = np.array(archive_F)

            # Check if new point is dominated
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if np.any(mask_le & mask_lt):
                return

            # Remove dominated points from archive
            mask_le2 = np.all(f <= F_arr, axis=1)
            mask_lt2 = np.any(f < F_arr, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)

            archive_X.append(x.copy())
            archive_F.append(f.copy())

        # Phase 1: Initialization - diverse random points
        n_init = 50
        X_init = np.zeros((n_init, dim))
        # 30 random interior points
        X_init[:30] = lo + span * np.random.random((30, dim))
        # 10 axis-aligned boundary points
        for i in range(10):
            x = lo + span * np.random.random(dim)
            axis = i % dim
            if i % 2 == 0:
                x[axis] = lo[axis]
            else:
                x[axis] = hi[axis]
            X_init[30 + i] = x
        # 10 corner points
        for i in range(10):
            X_init[40 + i] = np.where(np.random.random(dim) < 0.5, lo, hi)

        for x in X_init:
            evaluate_and_add(x)

        # Phase 2: Tchebycheff decomposition-based search
        # Create diverse weight vectors
        n_weights = 15
        weights = []
        # Systematic weights
        for i in range(n_weights):
            w1 = 0.1 + 0.8 * i / (n_weights - 1)
            w2 = 1.0 - w1
            weights.append(np.array([w1, w2]))
        # Add some random weights for diversity
        for _ in range(5):
            w1 = np.random.uniform(0.05, 0.95)
            weights.append(np.array([w1, 1.0 - w1]))

        # Assign each weight to a starting point from archive
        n_local_reserve = 30
        n_global = budget - used - n_local_reserve
        if n_global <= 0:
            n_local_reserve = max(0, budget - used)
            n_global = 0

        # For each weight, perform local search
        if len(archive_X) > 0 and n_global > 0:
            # Sort archive by first objective for diversity
            F_arr = np.array(archive_F)
            order = np.argsort(F_arr[:, 0])

            # Distribute evaluations across weights
            evals_per_weight = max(1, n_global // len(weights))

            for wi, w in enumerate(weights):
                if time.time() > deadline:
                    break
                if used >= budget - n_local_reserve:
                    break

                # Select starting point: use different archive points for different weights
                # Use the point that best matches this weight's preference
                # Score each archive point by Tchebycheff scalarization with this weight
                F_arr = np.array(archive_F)
                # Normalize objectives
                f_min = F_arr.min(axis=0)
                f_max = F_arr.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range < 1e-10] = 1.0
                F_norm = (F_arr - f_min) / f_range

                # Tchebycheff scalarization: min_i max_j w_j * (f_j - z_j) / z_range_j
                # Using normalized objectives
                scores = np.max(w[0] * F_norm[:, 0], axis=0) if n_obj == 2 else None
                # Actually compute properly
                if n_obj == 2:
                    scores = np.maximum(w[0] * F_norm[:, 0], w[1] * F_norm[:, 1])
                else:
                    scores = np.max(w[:, None] * F_norm, axis=0)

                # Pick the point with lowest Tchebycheff score
                best_idx = order[np.argmin(scores[order])]
                x0 = np.array(archive_X[best_idx])

                # Perform local search around x0
                for step in range(evals_per_weight):
                    if time.time() > deadline:
                        break
                    if used >= budget - n_local_reserve:
                        break

                    # Adaptive mutation step
                    progress = used / budget
                    scale = 0.3 * (1.0 - progress) + 0.05

                    # Generate candidate
                    child = x0 + scale * span * np.random.standard_normal(dim)
                    child = np.clip(child, lo, hi)

                    f_child = np.array(problem(child), dtype=np.float64)
                    used += 1

                    # Accept if it improves Tchebycheff score or is non-dominated
                    if len(archive_F) > 0:
                        F_arr = np.array(archive_F)
                        f_min = F_arr.min(axis=0)
                        f_max = F_arr.max(axis=0)
                        f_range = f_max - f_min
                        f_range[f_range < 1e-10] = 1.0

                        # Check if child is dominated
                        mask_le = np.all(F_arr <= f_child, axis=1)
                        mask_lt = np.any(F_arr < f_child, axis=1)
                        dominated = np.any(mask_le & mask_lt)

                        if not dominated:
                            evaluate_and_add(child)
                            x0 = child  # Update starting point
                        else:
                            # Check if child has better Tchebycheff score than x0
                            F_norm_child = (f_child - f_min) / f_range
                            if n_obj == 2:
                                score_child = max(w[0] * F_norm_child[0], w[1] * F_norm_child[1])
                                F_norm_x0 = (np.array(archive_F[archive_X.index(x0) if x0 in archive_X else 0]) - f_min) / f_range
                                score_x0 = max(w[0] * F_norm_x0[0], w[1] * F_norm_x0[1])
                            else:
                                score_child = np.max(w * F_norm_child)
                                score_x0 = np.max(w * F_norm_x0)

                            if score_child < score_x0:
                                evaluate_and_add(child)
                                x0 = child
                    else:
                        evaluate_and_add(child)
                        x0 = child

        # Phase 3: Final polish - refine best archive points
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            # Select diverse points from archive
            n_select = min(3, len(archive_X))
            order = np.argsort(F_arr[:, 0])
            indices = [order[0], order[len(order) // 2], order[-1]]
            indices = list(set(indices))[:n_select]

            for idx in indices:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])

                # Try a few small perturbations
                for _ in range(5):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    step = 0.02 * span
                    child = x0 + step * np.random.standard_normal(dim)
                    evaluate_and_add(child)

        if archive_X:
            return np.array(archive_X)
        return np.array([lo])
# EVOLVE-BLOCK-END
