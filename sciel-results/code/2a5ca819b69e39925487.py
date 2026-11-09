"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using archive-guided DE with adaptive scaling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 320.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # --- Initialization ---
        pop_size = 25

        # Generate structured initial points
        corners = []
        for i in range(2 ** dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(corner)

        structured_extra = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
            lo + 0.1 * span,
            lo + 0.9 * span,
            lo + 0.3 * span,
            lo + 0.7 * span,
            lo + 0.15 * span,
            lo + 0.85 * span,
        ]

        init_points = []
        # Sample corners
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_idx = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_idx])

        for sp in structured_extra:
            if len(init_points) < pop_size:
                init_points.append(sp)

        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Archive management ---
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            if not archive_F:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return
            A = np.array(archive_F)
            # Check if dominated by existing archive
            dominated = np.all(A <= f, axis=1) & np.any(A < f, axis=1)
            if np.any(dominated):
                return
            # Remove archive points dominated by f
            dominates = np.all(f <= A, axis=1) & np.any(f < A, axis=1)
            keep = ~dominates
            archive_X = [archive_X[i] for i in range(len(archive_X)) if keep[i]]
            archive_F = [archive_F[i] for i in range(len(archive_F)) if keep[i]]
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Fast non-dominated sorting ---
        def fast_nondominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            while fronts and not fronts[-1]:
                fronts.pop()

            # Crowding distances
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowd[idx] = np.inf
                    continue
                for m in range(n_obj):
                    sorted_idx = sorted(front, key=lambda k: F_mat[k][m])
                    crowd[sorted_idx[0]] = np.inf
                    crowd[sorted_idx[-1]] = np.inf
                    f_min = F_mat[sorted_idx[0]][m]
                    f_max = F_mat[sorted_idx[-1]][m]
                    if f_max - f_min < 1e-12:
                        continue
                    for k in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[k]
                        crowd[idx] += (F_mat[sorted_idx[k + 1]][m] - F_mat[sorted_idx[k - 1]][m]) / (f_max - f_min)

            return fronts, crowd

        # --- Main loop: DE-based multi-objective optimization ---
        main_budget = budget - max(30, budget // 7)

        # Track best hypervolume contribution for adaptive scaling
        def hypervolume_contribution(points, ref):
            """Approximate hypervolume using simple decomposition."""
            if not points:
                return 0.0
            points = np.array(points)
            # Filter points better than reference
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            points = points[valid]
            # Simple 3D hypervolume approximation
            n = len(points)
            if n == 1:
                return np.prod(ref - points[0])
            # Sort by first objective
            order = np.argsort(points[:, 0])
            points = points[order]
            vol = 0.0
            for i in range(n):
                # Volume contribution of point i
                if i == 0:
                    w0 = ref[0] - points[0, 0]
                else:
                    w0 = points[i - 1, 0] - points[i, 0]
                if w0 <= 0:
                    continue
                # For each subsequent point, check if it dominates in dims 1,2
                w1 = ref[1] - points[i, 1]
                w2 = ref[2] - points[i, 2]
                # Check if any later point has smaller values in both dims 1 and 2
                if i + 1 < n:
                    later = points[i + 1:]
                    mask = np.all(later < points[i, 1:], axis=1)
                    if np.any(mask):
                        best_later = later[mask]
                        w1 = min(w1, ref[1] - np.min(best_later[:, 1]))
                        w2 = min(w2, ref[2] - np.min(best_later[:, 2]))
                vol += w0 * max(0, w1) * max(0, w2)
            return vol

        best_hv = 0.0
        no_improve_count = 0
        F_SCALE = 0.8
        CR = 0.7

        while used < main_budget:
            if time.time() - start_time > DEADLINE:
                break

            # Create offspring using DE mutation
            offspring_X = []
            offspring_F = []

            # Sort population by non-dominated fronts for selection
            fronts, crowd = fast_nondominated_sort(F)

            # Build sorted index by front rank then crowding
            rank = np.zeros(len(F), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi
            sort_order = np.argsort(rank * 1000 - crowd)

            for _ in range(pop_size):
                if used >= main_budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                # Select parents from better fronts
                # Tournament: pick 3 random, select best by rank then crowding
                def tournament_select():
                    candidates = rng.choice(len(F), size=3, replace=False)
                    best = candidates[0]
                    for c in candidates[1:]:
                        if rank[c] < rank[best]:
                            best = c
                        elif rank[c] == rank[best] and crowd[c] > crowd[best]:
                            best = c
                    return best

                idx1 = tournament_select()
                idx2 = tournament_select()
                idx3 = tournament_select()

                # DE mutation: v = x1 + F * (x2 - x3)
                v = X[idx1] + F_SCALE * (X[idx2] - X[idx3])

                # Binomial crossover
                child = X[idx1].copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        child[j] = v[j]

                # Mutation
                if rng.random() < 1.0 / dim:
                    j = rng.integers(dim)
                    u = rng.random()
                    eta_m = 20.0
                    if u <= 1.0 / (eta_m + 1):
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1))
                    child[j] += delta * (hi[j] - lo[j])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            fronts, crowd = fast_nondominated_sort(combined_F)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    if remaining > 0:
                        sorted_front = sorted(front, key=lambda k: crowd[k], reverse=True)
                        for idx in sorted_front[:remaining]:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

            # Update archive
            for i in range(len(X)):
                update_archive(X[i], F[i])

            # Adaptive scaling based on archive improvement
            if archive_X:
                ref = np.array([1705.0, 12.0, 0.3])
                arch_arr = np.array(archive_F)
                valid_arch = np.all(arch_arr < ref, axis=1)
                if np.any(valid_arch):
                    hv = hypervolume_contribution(arch_arr[valid_arch], ref)
                    if hv > best_hv * 1.001:
                        best_hv = hv
                        no_improve_count = 0
                        F_SCALE = min(1.2, F_SCALE * 1.1)
                    else:
                        no_improve_count += 1
                        if no_improve_count > 5:
                            F_SCALE = max(0.4, F_SCALE * 0.85)
                            no_improve_count = 0

            # Diversity injection every 25 evaluations
            if used % 25 == 0 and used < main_budget:
                if time.time() - start_time > DEADLINE:
                    break
                if archive_X and rng.random() < 0.5:
                    base_arch = rng.integers(len(archive_X))
                    rand_x = archive_X[base_arch] + rng.normal(0, 0.3 * span, dim)
                else:
                    rand_x = lo + span * rng.random(dim)
                rand_x = np.clip(rand_x, lo, hi)
                rand_f = problem(rand_x)
                used += 1
                update_archive(rand_x, rand_f)
                # Replace worst individual
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # --- Polish phase: local search on archive points ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

            # Sort archive by sum of objectives
            sums = np.array([np.sum(af) for af in archive_F])
            top_n = min(15, len(archive_X))
            top_idx = np.argsort(sums)[:top_n]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Multi-scale local search
                sigmas = [0.1, 0.05, 0.025, 0.012]
                for sigma_scale in sigmas:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break

                    sigma = sigma_scale * span
                    n_trials = 3
                    for _ in range(n_trials):
                        if used >= budget:
                            break
                        if time.time() - start_time > DEADLINE:
                            break
                        perturbed = center + rng.normal(0, sigma, dim)
                        perturbed = np.clip(perturbed, lo, hi)

                        f_pert = problem(perturbed)
                        used += 1

                        if np.all(f_pert <= center_f) and np.any(f_pert < center_f):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert
                        elif not (np.all(center_f <= f_pert) and np.any(center_f < f_pert)):
                            if np.sum(f_pert) < np.sum(center_f) * 0.98:
                                update_archive(perturbed, f_pert)
                                center = perturbed
                                center_f = f_pert

        return X
# EVOLVE-BLOCK-END
