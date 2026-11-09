"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, budget)
        
        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (used for selection pressure)
        ref_point = np.array([1.1, 1.3, 1.1])

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_count[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[q] += 1
                if n_count[p] == 0:
                    rank[p] = 0
            i = 0
            front = [p for p in range(n) if rank[p] == 0]
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                front = next_front
                i += 1
            return rank

        def crowding_distance(F, rank, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(n_obj):
                vals = np.array([F[idx, m] for idx in indices])
                min_val = np.min(vals)
                max_val = np.max(vals)
                if max_val == min_val:
                    continue
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return dist

        while used < budget:
            if time.time() > deadline:
                break

            rank = fast_non_dominated_sort(F)
            
            # Create mating pool from first few fronts
            max_front = min(2, max(rank) + 1)
            mating_pool = [i for i in range(pop_size) if rank[i] < max_front]
            if len(mating_pool) == 0:
                mating_pool = list(range(pop_size))
            
            # Binary tournament selection
            def tournament():
                p1 = mating_pool[rng.integers(len(mating_pool))]
                p2 = mating_pool[rng.integers(len(mating_pool))]
                if rank[p1] < rank[p2]:
                    return p1
                elif rank[p2] < rank[p1]:
                    return p2
                else:
                    # Crowding distance for tie-breaking
                    front_idx = [i for i in range(pop_size) if rank[i] == rank[p1]]
                    cd = crowding_distance(F, rank, front_idx)
                    cd_map = {front_idx[k]: cd[k] for k in range(len(front_idx))}
                    return p1 if cd_map[p1] > cd_map[p2] else p2

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                p1 = tournament()
                p2 = tournament()
                # Crossover (SBX-like simple blend)
                if rng.random() < 0.9:
                    alpha = rng.random()
                    child = alpha * X[p1] + (1 - alpha) * X[p2]
                else:
                    child = X[p1].copy()
                
                # Mutation
                if rng.random() < 0.9:
                    mutation_rate = 1.0 / dim
                    mask = rng.random(dim) < mutation_rate
                    if np.any(mask):
                        child[mask] += 0.1 * (hi - lo) * rng.standard_normal(np.sum(mask))
                        child = np.clip(child, lo, hi)
                
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)
            
            rank_c = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            i = 0
            max_rank = max(rank_c) + 1
            while len(next_X) < pop_size and i <= max_rank:
                front_idx = [j for j in range(n_combined) if rank_c[j] == i]
                if len(next_X) + len(front_idx) <= pop_size:
                    for idx in front_idx:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    cd = crowding_distance(combined_F, rank_c, front_idx)
                    front_idx_sorted = np.argsort(-cd)
                    needed = pop_size - len(next_X)
                    for idx in front_idx_sorted[:needed]:
                        next_X.append(combined_X[front_idx[idx]])
                        next_F.append(combined_F[front_idx[idx]])
                i += 1

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
