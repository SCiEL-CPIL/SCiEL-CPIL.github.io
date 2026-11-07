"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Smaller population for more generations
        pop_size = 20

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def compute_crowding(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-14:
                    dist[:] = np.inf
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        while used < self.budget:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)
            
            # Precompute ranks and crowding distances
            rank = np.zeros(len(F), dtype=int)
            cd = np.zeros(len(F), dtype=np.float64)
            for idx, front in enumerate(fronts):
                for i in front:
                    rank[i] = idx
                cd_front = compute_crowding(F, front)
                for k, i in enumerate(front):
                    cd[i] = cd_front[k]

            n_offspring = pop_size
            offspring_X = []
            
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                # Tournament selection
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # Compare
                if rank[p1_idx] < rank[p2_idx]:
                    p1 = p1_idx
                elif rank[p2_idx] < rank[p1_idx]:
                    p1 = p2_idx
                else:
                    if cd[p1_idx] > cd[p2_idx]:
                        p1 = p1_idx
                    else:
                        p1 = p2_idx

                # Second parent
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                
                if rank[p3_idx] < rank[p4_idx]:
                    p2 = p3_idx
                elif rank[p4_idx] < rank[p3_idx]:
                    p2 = p4_idx
                else:
                    if cd[p3_idx] > cd[p4_idx]:
                        p2 = p3_idx
                    else:
                        p2 = p4_idx

                # Crossover
                if np.random.random() < 0.5:
                    child = (X[p1] + X[p2]) / 2.0
                else:
                    alpha = 0.5
                    lo_c = np.minimum(X[p1], X[p2]) - alpha * np.abs(X[p1] - X[p2])
                    hi_c = np.maximum(X[p1], X[p2]) + alpha * np.abs(X[p1] - X[p2])
                    child = lo_c + (hi_c - lo_c) * np.random.random(dim)
                
                # Mutate
                child = child + 0.1 * span * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += len(offspring_X)

            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Sort and select
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            # Precompute CD for selection
            comb_cd = np.zeros(len(combined_F), dtype=np.float64)
            for front in combined_fronts:
                cd_front = compute_crowding(combined_F, front)
                for k, i in enumerate(front):
                    comb_cd[i] = cd_front[k]

            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    remaining = pop_size - len(new_X)
                    # Sort by CD descending
                    order = np.argsort(-comb_cd[front])
                    selected = [front[i] for i in order[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
