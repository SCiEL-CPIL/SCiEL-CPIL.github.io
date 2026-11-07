import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0])

        # --- Hypervolume calculation (3D, O(N^2) via sweep) ---
        def hv_3d(points, ref):
            """Compute 3D hypervolume of points w.r.t. ref. points: (N,3), ref: (3,)"""
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)
            hv = 0.0
            # Precompute suffix min of obj2 and obj3 for fast 2D HV
            # For each i, we need the 2D HV of pts[i:] in (obj2, obj3) plane
            # 2D HV: sort by obj2, sum (y_{j+1}-y_j) * (ref_z - min_z_suffix)
            # We'll compute this incrementally
            for i in range(n):
                x_i = pts[i, 0]
                x_next = pts[i + 1, 0] if i < n - 1 else ref[0]
                dx = x_next - x_i
                if dx <= 0:
                    continue
                # 2D HV of pts[i:] in (obj2, obj3)
                sub = pts[i:, 1:3]
                sub = sub[np.all(sub < ref[1:3], axis=1)]
                if len(sub) == 0:
                    continue
                sub = sub[np.argsort(sub[:, 0])]
                k = len(sub)
                # Suffix min of obj3
                suffix_min = np.empty(k)
                suffix_min[k - 1] = sub[k - 1, 1]
                for j in range(k - 2, -1, -1):
                    suffix_min[j] = min(sub[j, 1], suffix_min[j + 1])
                area = 0.0
                for j in range(k):
                    y_next = sub[j + 1, 0] if j < k - 1 else ref[1]
                    dy = y_next - sub[j, 0]
                    if dy <= 0:
                        continue
                    hz = ref[2] - suffix_min[j]
                    if hz > 0:
                        area += dy * hz
                hv += dx * area
            return hv

        # --- Non-dominated sort ---
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            fronts = []
            current = [i for i in range(n) if n_dom[i] == 0]
            while current:
                fronts.append(current)
                next_front = []
                for i in current:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current = next_front
            return fronts

        # --- Crowding distance ---
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            F_sub = F[indices]
            for m in range(n_obj):
            # Sort by objective m
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range
            return cd

        # --- Initialize population ---
        pop_size = 20
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Main evolution loop ---
        # Use DE/rand/1 with binomial crossover
        # Track best hypervolume for stagnation detection
        best_hv = hv_3d(F, ref_point)
        stagnation_counter = 0
        stagnation_window = 30

        # Adaptive mutation factor
        F_mut = 0.8
        CR = 0.5

        while used < budget:
            if time.time() > deadline:
                break

            # Generate one offspring per iteration
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            v = X[r1].copy()
            j_rand = np.random.randint(0, n_dim)
            for j in range(n_dim):
                if np.random.random() < CR or j == j_rand:
                    v[j] = X[r1, j] + F_mut * (X[r2, j] - X[r3, j])
            v = np.clip(v, lo, hi)
            fv = problem(v)
            used += 1

            # Combine parent and offspring
            X_new = np.vstack([X, v])
            F_new = np.vstack([F, fv])

            # Selection: NSGA-II style with HV-based truncation
            fronts = non_dominated_sort(F_new)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    # Truncate using crowding distance
                    cd = crowding_distance(F_new, front)
                    front_arr = np.array(front)
                    order = np.argsort(cd, kind='stable')
                    selected.extend(front_arr[order[:remaining]].tolist())
                    break

            X = X_new[selected]
            F = F_new[selected]

            # Update best HV and stagnation
            current_hv = hv_3d(F, ref_point)
            if current_hv > best_hv + 1e-10:
                best_hv = current_hv
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adapt mutation factor based on stagnation
            if stagnation_counter > stagnation_window:
                # Diversify: increase mutation, reset some individuals
                F_mut = min(F_mut + 0.1, 1.5)
                CR = min(CR + 0.1, 0.9)
                # Replace worst individual with random
                fronts_cur = non_dominated_sort(F)
                if len(fronts_cur) > 1:
                    worst_front = fronts_cur[-1]
                    cd = crowding_distance(F, worst_front)
                    worst_idx = worst_front[np.argmin(cd)]
                    X[worst_idx] = np.random.uniform(lo, hi)
                    F[worst_idx] = problem(X[worst_idx])
                    used += 1
                stagnation_counter = 0

            # Gradually reduce mutation if improving
            if stagnation_counter == 0:
                F_mut = max(F_mut - 0.02, 0.5)
                CR = max(CR - 0.02, 0.3)

        return X
