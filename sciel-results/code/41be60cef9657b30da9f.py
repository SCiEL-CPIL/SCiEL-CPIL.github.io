# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using LHS + Nelder-Mead scalarization polishing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # --- Phase 1: Latin Hypercube initial sampling ---
        init_n = 50
        lhs = np.empty((init_n, dim))
        for d in range(dim):
            perm = rng.permutation(init_n)
            lhs[:, d] = (perm + rng.random(init_n)) / init_n
        X_init = lo + (hi - lo) * lhs
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = init_n

        archive_X = X_init.copy()
        archive_F = F_init.copy()

        # --- Helpers ---
        def non_dominated_mask(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                dominated = np.any(
                    np.all(F[:i] <= F[i], axis=1) & np.any(F[:i] < F[i], axis=1)
                )
                if not dominated:
                    dominated = np.any(
                        np.all(F[i+1:] <= F[i], axis=1) & np.any(F[i+1:] < F[i], axis=1)
                    )
                if dominated:
                    mask[i] = False
            return mask

        def is_dominated(point, archive_F):
            if len(archive_F) == 0:
                return False
            return np.any(np.all(archive_F <= point, axis=1) & np.any(archive_F < point, axis=1))

        def prune_archive(archive_X, archive_F):
            n = len(archive_F)
            if n <= 1:
                return archive_X, archive_F
            mask = non_dominated_mask(archive_F)
            return archive_X[mask], archive_F[mask]

        def crowding_distance(F):
            """Compute crowding distance for hypervolume-aware archive trimming."""
            n = len(F)
            if n <= 2:
                return np.ones(n)
            cd = np.zeros(n)
            for k in range(F.shape[1]):
                sorted_idx = np.argsort(F[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
                if f_range > 1e-12:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
            return cd

        # --- Phase 2: Adaptive local search with Nelder-Mead ---
        remaining = budget - used

        # Get initial non-dominated set
        nd_mask = non_dominated_mask(archive_F)
        nd_idx = np.where(nd_mask)[0]
        if len(nd_idx) == 0:
            nd_idx = np.array([np.argmin(np.sum(archive_F, axis=1))])

        sum_obj = np.sum(archive_F, axis=1)
        best_idx = nd_idx[np.argmin(sum_obj[nd_idx])]

        # Phase 2a: Random restarts around archive points (use ~40% of remaining)
        n_random = int(min(remaining * 0.4, remaining))
        for i in range(n_random):
            if time.time() > deadline - 60:
                break
            if used >= budget:
                break

            if i < n_random // 4:
                center = archive_X[best_idx]
            elif len(nd_idx) > 0 and rng.random() < 0.7:
                center = archive_X[nd_idx[rng.integers(len(nd_idx))]]
            else:
                center = archive_X[rng.integers(len(archive_X))]

            progress = i / max(1, n_random - 1)
            step = 0.12 * (1.0 - 0.6 * progress) + 0.02

            child = center + step * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            if not is_dominated(fc, archive_F):
                new_archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                new_archive_F = np.vstack([archive_F, fc.reshape(1, -1)])
                archive_X, archive_F = prune_archive(new_archive_X, new_archive_F)

                nd_mask = non_dominated_mask(archive_F)
                nd_idx = np.where(nd_mask)[0]
                if len(nd_idx) > 0:
                    sum_obj_new = np.sum(archive_F, axis=1)
                    best_idx = nd_idx[np.argmin(sum_obj_new[nd_idx])]

        # Phase 2b: Nelder-Mead scalarization polishing (use remaining budget)
        if time.time() < deadline - 30 and used < budget:
            nd_mask = non_dominated_mask(archive_F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                sum_obj = np.sum(archive_F, axis=1)
                nd_sorted = nd_idx[np.argsort(sum_obj[nd_idx])]
                n_polish = min(6, len(nd_sorted), max(0, budget - used))

                for k in range(n_polish):
                    if time.time() > deadline - 20:
                        break
                    if used >= budget:
                        break

                    start_x = archive_X[nd_sorted[k]].copy()

                    # Multiple weight directions for diversity
                    for trial in range(2):
                        if used >= budget:
                            break
                        if time.time() > deadline - 15:
                            break

                        weights = rng.dirichlet(np.ones(n_obj))

                        # Nelder-Mead with explicit evaluation counting
                        simplex = np.empty((dim + 1, dim))
                        simplex[0] = start_x
                        for j in range(1, dim + 1):
                            simplex[j] = start_x.copy()
                            simplex[j, j - 1] += 0.05 * (hi[j - 1] - lo[j - 1])
                            simplex[j] = np.clip(simplex[j], lo, hi)

                        simplex_F = np.array([problem(simplex[j]) for j in range(dim + 1)])
                        used += dim + 1

                        # Track best
                        best_local_F = simplex_F.copy()
                        best_local_x = simplex.copy()

                        max_iter = 20
                        for it in range(max_iter):
                            if used >= budget - 5:
                                break
                            if time.time() > deadline - 10:
                                break

                            # Sort simplex by weighted sum
                            wsums = simplex_F @ weights
                            order = np.argsort(wsums)
                            simplex = simplex[order]
                            simplex_F = simplex_F[order]

                            # Reflection
                            centroid = simplex[1:].mean(axis=0)
                            x_ref = centroid + 1.0 * (centroid - simplex[0])
                            x_ref = np.clip(x_ref, lo, hi)
                            f_ref = problem(x_ref)
                            used += 1

                            if wsums[0] < simplex_F[0] @ weights < wsums[-1]:
                                simplex[0] = x_ref
                                simplex_F[0] = f_ref
                            elif simplex_F[0] @ weights <= wsums[0]:
                                # Expansion
                                x_exp = centroid + 2.0 * (centroid - simplex[0])
                                x_exp = np.clip(x_exp, lo, hi)
                                f_exp = problem(x_exp)
                                used += 1
                                if f_exp @ weights < wsums[0]:
                                    simplex[0] = x_exp
                                    simplex_F[0] = f_exp
                                else:
                                    simplex[0] = x_ref
                                    simplex_F[0] = f_ref
                            else:
                                # Contraction
                                x_con = centroid + 0.5 * (simplex[0] - centroid)
                                x_con = np.clip(x_con, lo, hi)
                                f_con = problem(x_con)
                                used += 1
                                if f_con @ weights < wsums[0]:
                                    simplex[0] = x_con
                                    simplex_F[0] = f_con
                                else:
                                    # Shrink
                                    for j in range(1, dim + 1):
                                        simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                        simplex[j] = np.clip(simplex[j], lo, hi)
                                        simplex_F[j] = problem(simplex[j])
                                        used += 1

                            # Update local best
                            cur_best = simplex_F @ weights
                            best_i = np.argmin(cur_best)
                            if cur_best[best_i] < best_local_F @ weights:
                                best_local_F = simplex_F[best_i]
                                best_local_x = simplex[best_i]

                        # Add best local result to archive
                        if not is_dominated(best_local_F, archive_F):
                            new_archive_X = np.vstack([archive_X, best_local_x.reshape(1, -1)])
                            new_archive_F = np.vstack([archive_F, best_local_F.reshape(1, -1)])
                            archive_X, archive_F = prune_archive(new_archive_X, new_archive_F)

        return archive_X

# EVOLVE-BLOCK-END
