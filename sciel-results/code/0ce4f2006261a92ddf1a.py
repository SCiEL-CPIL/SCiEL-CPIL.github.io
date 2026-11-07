import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)
        
        pop_size = 30
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominating[i] += 1
                if n_dominating[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
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

        def select_parents(F, X, n_parents):
            fronts = fast_non_dominated_sort(F)
            parents = []
            for front in fronts:
                if len(parents) + len(front) <= n_parents:
                    parents.extend(front)
                else:
                    remaining = n_parents - len(parents)
                    if remaining > 0:
                        cd = crowding_distance(F, front)
                        sorted_front = [front[j] for j in np.argsort(-cd)]
                        parents.extend(sorted_front[:remaining])
                    break
            return parents

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parents
            n_parents = pop_size
            parent_indices = select_parents(F, X, n_parents)
            
            # Generate children
            n_children = min(pop_size, self.budget - used)
            X_c = np.zeros((n_children, self.dim))
            
            for i in range(n_children):
                # Pick two distinct parents
                p1_idx = parent_indices[rng.integers(len(parent_indices))]
                p2_idx = parent_indices[rng.integers(len(parent_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Uniform Crossover
                mask = rng.random(self.dim) < 0.5
                child = np.where(mask, p1, p2)
                
                # Mutation
                progress = used / self.budget
                step = (1.0 - 0.5 * progress) * 0.15 * (hi - lo)
                child = child + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                X_c[i] = child
            
            F_c = np.array([problem(x) for x in X_c])
            used += n_children
            
            # Combine and Select
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
            
        return X
