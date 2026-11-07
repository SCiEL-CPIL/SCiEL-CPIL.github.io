"""Multi-objective optimizer on WFG4 using NSGA-II with vectorized operations."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Initialize population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        all_F = [tuple(f) for f in F]  # record all evaluated objective vectors

        # Dedup set for objective space
        seen = set(all_F)

        def time_left():
            return deadline - time.time()

        def vectorized_dominates(F_new, F_arr):
            """Return boolean array: True where F_new dominates F_arr[i]."""
            # F_new: (n_obj,), F_arr: (n, n_obj)
            all_le = np.all(F_new <= F_arr, axis=1)
            any_lt = np.any(F_new < F_arr, axis=1)
            return all_le & any_lt

        def non_dominated_sort(F_arr):
            """Vectorized non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F_arr)
            if n == 0:
                return []
            fronts = []
            remaining = set(range(n))
            S = {i: [] for i in range(n)}
            nd = [0] * n

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1

            front = [i for i in range(n) if nd[i] == 0]
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(self.n_obj):
                vals = np.array([F_arr[i, m] for i in indices])
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 1e-15:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / val_range
            return cd

        def tournament_select(F_arr, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F_arr)
            # Assign ranks
            rank = [0] * n
            for r, front in enumerate(fronts):
                for i in front:
                    rank[i] = r

            # Crowding distance per front
            cd = [0.0] * n
            for front in fronts:
                cd_vals = crowding_distance(F_arr, front)
                for idx, dist in zip(front, cd_vals):
                    cd[idx] = dist

            # Tournament
            candidates = rng.integers(0, n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Check if we should do local refinement in the last portion
            remaining = self.budget - used
            if remaining < 10:
                # Final polish: local search around best points
                fronts = non_dominated_sort(F)
                if fronts:
                    front0 = fronts[0]
                    parent_idx = front0[rng.integers(len(front0))]
                else:
                    parent_idx = int(np.argmin(np.sum(F, axis=1)))

                parent = X[parent_idx]
                # Small step local search
                step = 0.02 * (hi - lo)
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = np.array(problem(child), dtype=np.float64)
                used += 1
                key = tuple(fc)
                if key not in seen:
                    seen.add(key)
                    all_F.append(key)
                X[parent_idx] = child
                F[parent_idx] = fc
                continue

            # NSGA-II step
            fronts = non_dominated_sort(F)

            # Generate offspring via tournament + mutation
            n_offspring = min(10, self.budget - used)
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)
                parent = X[p1]

                # Adaptive step size: larger early, smaller late
                progress = used / self.budget
                base_step = 0.3 * (1.0 - 0.7 * progress)
                step = base_step * (hi - lo)
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = np.array(problem(child), dtype=np.float64)
                used += 1
                key = tuple(fc)
                if key not in seen:
                    seen.add(key)
                    all_F.append(key)

                # Replace worst in population
                dom_mask = vectorized_dominates(fc, F)
                if np.any(dom_mask):
                    # Replace a dominated solution
                    dom_indices = np.where(dom_mask)[0]
                    worst_idx = dom_indices[rng.integers(len(dom_indices))]
                else:
                    # Replace the one with worst crowding distance in last front
                    last_front = fronts[-1] if fronts else list(range(len(F)))
                    cd_vals = crowding_distance(F, last_front)
                    worst_in_front = last_front[int(np.argmin(cd_vals))]
                    worst_idx = worst_in_front

                X[worst_idx] = child
                F[worst_idx] = fc

                # Re-sort only if needed (every few iterations)
                if used % 5 == 0:
                    fronts = non_dominated_sort(F)

        return all_F
