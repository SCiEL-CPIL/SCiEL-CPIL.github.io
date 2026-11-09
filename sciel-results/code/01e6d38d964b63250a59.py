"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the evaluator's seeded RNG
        rng = np.random

        # Phase 1: Latin Hypercube Sampling initialization
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(pop_size) + rng.random(pop_size)) / pop_size

        used = 0
        for i in range(pop_size):
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
            n = len(F_arr)
            if n == 0:
                return []
            
            # Vectorized dominance check
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = []
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            F_front = F_arr[front_indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_front[sorted_idx[i+1], m] - F_front[sorted_idx[i-1], m]) / f_range
            
            return cd

        def tournament_select(F_arr, fronts, cd_map, k=2):
            """Tournament selection based on rank then crowding distance."""
            n = len(F_arr)
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            
            # Precompute rank for each candidate
            rank = {}
            for r, f in enumerate(fronts):
                for idx in f:
                    rank[idx] = r
            
            for c in candidates[1:]:
                if rank[best] < rank[c]:
                    continue
                elif rank[c] < rank[best]:
                    best = c
                else:
                    # Same rank, compare crowding distance
                    cd_b = cd_map.get(rank[best], {}).get(best, 0)
                    cd_c = cd_map.get(rank[c], {}).get(c, 0)
                    if cd_c > cd_b:
                        best = c
            return best

        # Phase 2: NSGA-II-style evolution
        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts and crowding distances
            fronts = fast_non_dominated_sort(F)
            cd_map = {}
            for fi, front in enumerate(fronts):
                cd_vals = crowding_distance(F, front)
                cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            n_off = 0

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select parents via tournament
                p1 = tournament_select(F, fronts, cd_map)
                p2 = tournament_select(F, fronts, cd_map)

                # Crossover (BLX-alpha style)
                if rng.random() < 0.9:
                    alpha = 0.5
                    for d in range(dim):
                        if p1 != p2:
                            lo_c = min(X[p1, d], X[p2, d]) - alpha * abs(X[p1, d] - X[p2, d])
                            hi_c = max(X[p1, d], X[p2, d]) + alpha * abs(X[p1, d] - X[p2, d])
                            offspring_X[n_off, d] = rng.uniform(max(lo[d], lo_c), min(hi[d], hi_c))
                        else:
                            offspring_X[n_off, d] = X[p1, d]
                else:
                    offspring_X[n_off, :] = X[p1, :]

                # Adaptive mutation: mix of Gaussian and Cauchy-like perturbations
                progress = used / budget
                scale = 0.15 * (1.0 - 0.5 * progress)
                
                if rng.random() < 0.7:
                    noise = rng.standard_normal(dim)
                else:
                    noise = rng.standard_normal(dim) / (0.3 + 0.7 * np.abs(rng.standard_normal(dim)))
                    noise = np.clip(noise, -3, 3)
                
                offspring_X[n_off, :] += scale * (hi - lo) * noise
                offspring_X[n_off, :] = np.clip(offspring_X[n_off, :], lo, hi)

                offspring_F[n_off] = problem(offspring_X[n_off, :])
                used += 1
                n_off += 1

            if n_off == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X[:pop_size], offspring_X[:n_off]])
            combined_F = np.vstack([F[:pop_size], offspring_F[:n_off]])

            # Non-dominated sort the combined set
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_cd_map = {}
            for fi, front in enumerate(combined_fronts):
                cd_vals = crowding_distance(combined_F, front)
                combined_cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            # Select next population
            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            next_count = 0
            for fi, front in enumerate(combined_fronts):
                if next_count + len(front) <= pop_size:
                    for idx in front:
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                else:
                    # Need to select some from this front based on crowding distance
                    remaining = pop_size - next_count
                    if remaining <= 0:
                        break
                    cd_vals = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd_vals[i], reverse=True)
                    for i in range(min(remaining, len(sorted_front))):
                        idx = front[sorted_front[i]]
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                if next_count >= pop_size:
                    break

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END
