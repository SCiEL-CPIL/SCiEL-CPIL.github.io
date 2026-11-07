"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: NSGA-II style with adaptive mutation for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 30

        # Initialize population
        X = np.empty((pop_size, dim))
        # Use a mix: some random, some near center, some spread
        n1 = pop_size // 3
        n2 = pop_size // 3
        n3 = pop_size - n1 - n2

        X[:n1] = lo + span * np.random.rand(n1, dim)
        if n2 > 0:
            X[n1:n1+n2] = lo + span * np.clip(0.5 + 0.2 * np.random.randn(n2, dim), 0, 1)
        if n3 > 0:
            # LHS-like spread
            for i in range(n3):
                X[n1+n2+i] = lo + span * ((i + 0.5) / n3 + 0.1 * np.random.randn(dim))
            X[n1+n2:] = np.clip(X[n1+n2:], lo, hi)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Helper: fast non-dominated sorting
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dominated = np.zeros(n, dtype=int)
            front = [np.where(np.all(~(np.all(F_arr <= F_arr[i], axis=1) & np.any(F_arr < F_arr[i], axis=1) | (F_arr[i] == F_arr[i])), axis=1))[0].tolist()] if False else None
            
            # Simpler implementation
            fronts = []
            remaining = set(range(n))
            while remaining:
                current_front = []
                for i in remaining:
                    dominated_by_any = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated_by_any = True
                            break
                    if not dominated_by_any:
                        current_front.append(i)
                if not current_front:
                    # Fallback: put all remaining in one front
                    current_front = list(remaining)
                fronts.append(current_front)
                remaining -= set(current_front)
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                min_val = np.min(vals)
                max_val = np.max(vals)
                if max_val - min_val < 1e-12:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n-1):
                    dist[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / (max_val - min_val)
            return dist

        def select_parents(F_arr, fronts):
            """Select two parents using tournament selection based on fronts and crowding."""
            # Assign ranks
            rank = np.zeros(len(F_arr), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Crowding distance per front
            crowd = np.zeros(len(F_arr))
            for front in fronts:
                if len(front) > 2:
                    cd = crowding_distance(F_arr, np.array(front))
                    for k, idx in enumerate(front):
                        crowd[idx] = cd[k]
            
            # Tournament selection
            def tournament():
                i = np.random.randint(len(F_arr))
                j = np.random.randint(len(F_arr))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if crowd[i] > crowd[j] else j
            
            p1 = tournament()
            p2 = tournament()
            if p1 == p2:
                p2 = np.random.randint(len(F_arr))
            return p1, p2

        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents
            p1, p2 = select_parents(F, fronts)

            # Crossover: blend
            alpha = np.random.rand(dim)
            child = alpha * X[p1] + (1 - alpha) * X[p2]

            # Adaptive mutation
            progress = used / budget
            step = 0.3 * (1 - progress) + 0.02
            noise = np.random.randn(dim) * step * span
            child = child + noise
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Combine and select
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])
            
            # Sort and select top pop_size
            fronts_new = fast_non_dominated_sort(F_new)
            
            selected = []
            for front in fronts_new:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    need = pop_size - len(selected)
                    if len(front) > need:
                        cd = crowding_distance(F_new, np.array(front))
                        order = np.argsort(-cd)
                        selected.extend([front[k] for k in order[:need]])
                    else:
                        selected.extend(front)
                    break
            
            X = X_new[selected]
            F = F_new[selected]

        return X
# EVOLVE-BLOCK-END
