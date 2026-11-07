"""Improved multi-objective optimizer for ZDT3 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        pop_size = min(40, self.budget // 2)
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sort, return list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominated
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = F.shape[1]
            
            for k in range(m):
                sorted_idx = np.argsort(F[front_indices, k])
                sorted_idx = [front_indices[i] for i in sorted_idx]
                cd_pos = np.zeros(n)
                cd_pos[0] = np.inf
                cd_pos[-1] = np.inf
                f_range = F[front_indices[sorted_idx[-1]], k] - F[front_indices[sorted_idx[0]], k]
                if f_range > 0:
                    for i in range(1, n - 1):
                        idx = front_indices[sorted_idx[i]]
                        cd_pos[i] += (F[front_indices[sorted_idx[i+1]], k] - F[front_indices[sorted_idx[i-1]], k]) / f_range
                cd += cd_pos
            
            return cd

        # Main evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Select parents using tournament selection
            def tournament_select():
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                # Find front ranks
                rank_i = rank_j = 0
                for fi, front in enumerate(fronts):
                    if i in front:
                        rank_i = fi
                    if j in front:
                        rank_j = fi
                    if rank_i > 0 and rank_j > 0:
                        break
                if rank_i < rank_j:
                    return i
                elif rank_j < rank_i:
                    return j
                else:
                    # Same front, use crowding distance
                    cd = crowding_distance(F, fronts[rank_i])
                    idx_i = list(fronts[rank_i]).index(i)
                    idx_j = list(fronts[rank_i]).index(j)
                    return i if cd[idx_i] > cd[idx_j] else j
            
            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                if used + len(offspring_F) >= self.budget:
                    break
                    
                p1 = tournament_select()
                p2 = tournament_select()
                
                # SBX crossover
                if rng.random() < 0.9:
                    beta = rng.random()
                    if beta < 0.5:
                        beta = 1.0 / (2 * beta)
                    else:
                        beta = 1.0 / (2 * (1 - beta))
                    
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child1[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                            child2[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                        else:
                            child1[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                            child2[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    if used + len(offspring_F) + 1 < self.budget:
                        offspring_X.append(child1)
                        offspring_F.append(problem(child1))
                        used += 1
                    if used + len(offspring_F) + 1 < self.budget:
                        offspring_X.append(child2)
                        offspring_F.append(problem(child2))
                        used += 1
                else:
                    # Just copy
                    child = X[p1].copy()
                    child = np.clip(child, lo, hi)
                    offspring_X.append(child)
                    offspring_F.append(problem(child))
                    used += 1
            
            if len(offspring_F) == 0:
                break
            
            # Combine parents and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort on combined population
            fronts_combined = fast_non_dominated_sort(F_combined)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_combined:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[list(front).index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
            
            if len(new_X) == 0:
                break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
