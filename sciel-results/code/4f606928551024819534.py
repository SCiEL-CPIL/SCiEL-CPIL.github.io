"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 20
        if self.budget < pop_size:
            pop_size = max(1, self.budget)
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    front.append(i)
            
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort([F[i, m] for i in indices])
                sorted_indices = [indices[i] for i in sorted_idx]
                
                min_val = F[sorted_indices[0], m]
                max_val = F[sorted_indices[-1], m]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                    
                dist[0] = np.inf
                dist[-1] = np.inf
                
                for i in range(1, n - 1):
                    dist[i] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / range_val
            return dist

        def blx_alpha(x1, x2, alpha=0.5):
            lower = np.minimum(x1, x2)
            upper = np.maximum(x1, x2)
            
            spread = alpha * (upper - lower)
            lower -= spread
            upper += spread
            
            lower = np.maximum(lower, lo)
            upper = np.minimum(upper, hi)
            
            x_child = lower + (upper - lower) * np.random.random(self.dim)
            return x_child

        def gaussian_mutation(x, sigma=0.1):
            x_mut = x + np.random.normal(0, sigma, self.dim)
            return np.clip(x_mut, lo, hi)

        while used < self.budget:
            fronts = fast_non_dominated_sort(F)
            front_dists = []
            for f_idx in fronts:
                dists = crowding_distance(F, f_idx)
                front_dists.append(dists)
                
            def select_one():
                i = np.random.randint(pop_size)
                j = np.random.randint(pop_size)
                
                rank_i = -1
                rank_j = -1
                dist_i = -1
                dist_j = -1
                
                for r, f in enumerate(fronts):
                    if i in f:
                        rank_i = r
                        pos_i = f.index(i)
                        dist_i = front_dists[r][pos_i]
                        break
                for r, f in enumerate(fronts):
                    if j in f:
                        rank_j = r
                        pos_j = f.index(j)
                        dist_j = front_dists[r][pos_j]
                        break
                        
                if rank_i < rank_j:
                    return i
                elif rank_j < rank_i:
                    return j
                else:
                    if dist_i > dist_j:
                        return i
                    else:
                        return j

            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= self.budget:
                    break
                    
                p1_idx = select_one()
                p2_idx = select_one()
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                x_child = blx_alpha(x1, x2)
                x_child = gaussian_mutation(x_child)
                
                f_child = problem(x_child)
                used += 1
                
                offspring_X.append(x_child)
                offspring_F.append(f_child)
                
            if not offspring_X:
                break
                
            total_X = np.vstack([X, np.array(offspring_X)])
            total_F = np.vstack([F, np.array(offspring_F)])
            
            total_fronts = fast_non_dominated_sort(total_F)
            
            new_X = []
            new_F = []
            
            for f_idx, f in enumerate(total_fronts):
                if len(new_X) + len(f) <= pop_size:
                    for idx in f:
                        new_X.append(total_X[idx])
                        new_F.append(total_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    dists = crowding_distance(total_F, f)
                    sorted_indices = np.argsort(-dists)
                    for k in range(remaining):
                        idx = f[sorted_indices[k]]
                        new_X.append(total_X[idx])
                        new_F.append(total_F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
