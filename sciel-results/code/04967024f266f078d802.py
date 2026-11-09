"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard, crowding-based selection, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # 5s safety margin before 300s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 2  # 2s safety margin

        # Phase 1: Latin Hypercube Sampling for initial population
        pop_size = min(50, budget // 2)
        used = 0

        def lhs_sample(n, d):
            """Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points

        X = lo + span * lhs_sample(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: compute non-dominated front
        def non_dominated_indices(F):
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        # Helper: crowding distance for a set of points
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(F.shape[1]):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F[order[-1], m] - F[order[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
            return cd

        # Main evolutionary loop
        while used < budget and time_ok():
            # Find non-dominated set
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Use crowding distance to select parent from non-dominated set
            if len(nd_idx) > 1:
                cd = crowding_distance(F[nd_idx])
                # Select parent with highest crowding distance (most diverse)
                parent_local = nd_idx[np.argmax(cd)]
            else:
                parent_local = nd_idx[0]

            parent = X[parent_local].copy()

            # Adaptive mutation step size
            gen = used - pop_size
            total_gens = max(1, budget - pop_size)
            step = max(0.03, 0.15 * (1.0 - gen / total_gens))
            child = parent + step * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Check if child is non-dominated
            is_child_nd = True
            for i in range(len(F)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    is_child_nd = False
                    break

            if is_child_nd:
                # Replace a dominated point if exists, else replace worst
                dominated_mask = np.array([
                    np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
                    for i in range(len(F))
                ])
                if np.any(dominated_mask):
                    replace_idx = np.where(dominated_mask)[0][rng.integers(np.sum(dominated_mask))]
                else:
                    replace_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Child is dominated; only replace if it improves the worst point
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    replace_idx = worst_idx
                else:
                    continue  # Don't replace; skip this child

            X[replace_idx] = child
            F[replace_idx] = fc

        # Phase 2: Local refinement using remaining budget
        remaining = budget - used
        if remaining > 5 and time_ok():
            # Find current non-dominated front
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select up to 4 points from the front for local optimization
            n_refine = min(4, len(nd_idx), remaining // 4)
            if n_refine > 0:
                # Select diverse points from front using crowding distance
                if len(nd_idx) > n_refine:
                    cd = crowding_distance(F[nd_idx])
                    # Greedy selection by crowding distance
                    selected = []
                    remaining_nd = list(nd_idx)
                    for _ in range(n_refine):
                        if not remaining_nd:
                            break
                        cd_sub = crowding_distance(F[remaining_nd])
                        best_local = np.argmax(cd_sub)
                        selected.append(remaining_nd[best_local])
                        remaining_nd.pop(best_local)
                else:
                    selected = list(nd_idx)

                for idx in selected:
                    if not time_ok() or used >= budget:
                        break

                    x0 = X[idx].copy()
                    f0 = F[idx].copy()

                    # Try to improve using achievement scalarizing function with different weight vectors
                    # Use a few different weight combinations to explore the front
                    weight_sets = [
                        [0.5, 0.5],
                        [0.7, 0.3],
                        [0.3, 0.7],
                        [0.9, 0.1],
                        [0.1, 0.9],
                    ]

                    for weights in weight_sets:
                        if used >= budget or not time_ok():
                            break

                        # Achievement scalarizing function: minimize max(w_i * (f_i - z_i))
                        # where z is the ideal point (min of each objective)
                        z = np.min(F, axis=0)

                        best_x = x0.copy()
                        best_f = f0.copy()
                        best_val = np.max(weights * (f0 - z))

                        # Simple coordinate descent / local search
                        for _iter in range(3):
                            if used >= budget or not time_ok():
                                break

                            improved = False
                            for d in range(dim):
                                for direction in [-1, 1]:
                                    step_size = 0.02 * span[d]
                                    x_try = best_x.copy()
                                    x_try[d] += direction * step_size
                                    x_try = np.clip(x_try, lo, hi)

                                    f_try = problem(x_try)
                                    used += 1
                                    if used >= budget:
                                        break

                                    val = np.max(weights * (f_try - z))
                                    if val < best_val - 1e-10:
                                        best_x = x_try
                                        best_f = f_try
                                        best_val = val
                                        improved = True

                            if not improved:
                                break

                        # Check if this improves the Pareto front
                        dominates_orig = np.all(best_f <= f0) and np.any(best_f < f0)
                        if dominates_orig:
                            X[idx] = best_x
                            F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
