import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)
        
        pop_size = 16
        
        # Initialize with diverse spread
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of all non-dominated points seen
        archive_X = X.copy()
        archive_F = F.copy()
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_doming = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doming[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doming[i] += 1
                if n_doming[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_doming[j] -= 1
                        if n_doming[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def update_archive(X_new, F_new):
            nonlocal archive_X, archive_F
            # Combine and find non-dominated
            X_all = np.vstack([archive_X, X_new])
            F_all = np.vstack([archive_F, F_new])
            fronts = fast_non_dominated_sort(F_all)
            if fronts:
                front0 = fronts[0]
                archive_X = X_all[front0]
                archive_F = F_all[front0]

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parents via tournament on non-dominated fronts
            fronts = fast_non_dominated_sort(F)
            # Prefer first front, then second if needed
            parent_pool = []
            for front in fronts:
                parent_pool.extend(front)
                if len(parent_pool) >= pop_size:
                    break
            if len(parent_pool) < 2:
                parent_pool = list(range(len(F)))
            
            n_parents = min(len(parent_pool), pop_size)
            
            # Generate children
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break
            
            X_c = np.zeros((n_children, self.dim))
            
            for i in range(n_children):
                # Pick two parents for crossover
                p1_idx = parent_pool[rng.integers(n_parents)]
                p2_idx = parent_pool[rng.integers(n_parents)]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Uniform crossover
                mask = rng.random(self.dim) < 0.5
                child = np.where(mask, p1, p2)
                
                # Adaptive mutation: larger early, smaller late
                progress = used / self.budget
                # Start with 0.25 range, shrink to 0.05
                step_scale = 0.25 - 0.20 * progress
                step = step_scale * (hi - lo)
                child = child + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                X_c[i] = child
            
            F_c = np.array([problem(x) for x in X_c])
            used += n_children
            
            # Update archive with new children
            update_archive(X_c, F_c)
            
            # Combine parent and child populations for selection
            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            
            fronts = fast_non_dominated_sort(F_all)
            
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining > 0:
                        cd = crowding_distance(F_all, front)
                        sorted_front = [front[j] for j in np.argsort(-cd)]
                        new_indices.extend(sorted_front[:remaining])
                    break
            
            X = X_all[new_indices]
            F = F_all[new_indices]
        
        return archive_X
