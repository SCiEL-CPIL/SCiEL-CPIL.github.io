# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using NSGA-II strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = 40
        if budget < pop_size:
            pop_size = budget

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_nondominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            if not fronts[-1]:
                fronts.pop()
            return fronts

        def crowding_distance(F_mat, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                range_val = F_sub[order[-1], k] - F_sub[order[0], k]
                if range_val > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / range_val
            return cd

        def select_next_generation(F_mat, X_mat, target_size):
            fronts = fast_nondominated_sort(F_mat)
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= target_size:
                    selected_indices.extend(front)
                else:
                    # Need to select from this front
                    remaining = target_size - len(selected_indices)
                    cd = crowding_distance(F_mat, front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    selected_indices.extend([front[i] for i in order[:remaining]])
                    break
            return selected_indices

        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            # Select parents from the current non-dominated set
            fronts = fast_nondominated_sort(F)
            # Use the first front for parent selection
            parents_idx = np.array(fronts[0]) if fronts[0] else np.arange(pop_size)
            
            # If front is small, add some from next front to ensure diversity in parents
            if len(parents_idx) < pop_size // 2 and len(fronts) > 1:
                parents_idx = np.concatenate([parents_idx, np.array(fronts[1])])

            offspring_X = np.empty_like(X)
            for i in range(pop_size):
                # Tournament selection or random from parents
                p1_idx = parents_idx[rng.integers(len(parents_idx))]
                p2_idx = parents_idx[rng.integers(len(parents_idx))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # Blend Crossover
                if rng.random() < 0.8:
                    alpha = rng.random()
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()

                # Mutation
                # Adaptive step size
                progress = used / budget
                base_step = 0.1 * (1.0 - 0.5 * progress)
                step = base_step * (0.5 + rng.random())
                mask = rng.random(dim) < 0.3
                if mask.any():
                    child[mask] += step * (hi - lo)[mask] * rng.standard_normal(int(mask.sum()))
                
                child = np.clip(child, lo, hi)
                offspring_X[i] = child

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Select next generation
            next_indices = select_next_generation(F_combined, X_combined, pop_size)
            
            X = X_combined[next_indices]
            F = F_combined[next_indices]

        return X
# EVOLVE-BLOCK-END
