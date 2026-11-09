"""Rewritten program using NSGA-II style multi-objective evolutionary optimization for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(50, max(10, budget // 5))
        
        # Initial population: random within bounds
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
                    if not fronts:
                        fronts.append([])
                    fronts[0].append(i)

            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            m = len(indices)
            if m == 0:
                return []
            if m == 1:
                return [float('inf')]
            
            cd = [0.0] * m
            n_obj = F.shape[1]
            for o in range(n_obj):
                obj_vals = F[indices, o]
                sorted_idx = np.argsort(obj_vals)
                cd_sorted = [0.0] * m
                cd_sorted[0] = float('inf')
                cd_sorted[-1] = float('inf')
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range == 0:
                    continue
                for i in range(1, m - 1):
                    cd_sorted[i] = (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / obj_range
                for i in range(m):
                    cd[i] += cd_sorted[i]
            return cd

        while used < budget:
            # Create offspring population of same size
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            for _ in range(n_offspring):
                # Select parents using tournament selection based on rank and crowding distance
                fronts = fast_non_dominated_sort(F)
                rank = np.zeros(len(F), dtype=int)
                for i, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = i
                
                # Compute crowding distances for each front
                cd = np.zeros(len(F))
                for front in fronts:
                    if len(front) > 0:
                        cd_vals = crowding_distance(F, front)
                        for k, idx in enumerate(front):
                            cd[idx] = cd_vals[k]
                
                # Tournament selection
                def tournament_select():
                    i1 = rng.integers(len(F))
                    i2 = rng.integers(len(F))
                    if rank[i1] < rank[i2]:
                        return i1
                    elif rank[i2] < rank[i1]:
                        return i2
                    else:
                        if cd[i1] > cd[i2]:
                            return i1
                        else:
                            return i2
                
                p1 = tournament_select()
                p2 = tournament_select()
                
                # SBX crossover
                if rng.random() < 0.9:
                    eta_c = 20
                    x1 = X[p1]
                    x2 = X[p2]
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            if abs(x1[d] - x2[d]) < 1e-14:
                                child1[d] = x1[d]
                                child2[d] = x2[d]
                            else:
                                u = rng.random()
                                if u <= 0.5:
                                    beta = (2 * u) ** (1 / (eta_c + 1))
                                else:
                                    beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                        else:
                            child1[d] = x2[d]
                            child2[d] = x1[d]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.extend([child1, child2])
                    offspring_F.extend([problem(child1), problem(child2)])
                    used += 2
                else:
                    # Just copy
                    offspring_X.append(X[p1].copy())
                    offspring_F.append(problem(X[p1].copy()))
                    used += 1
                    if used >= budget:
                        break
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)]) if len(offspring_X) > 0 else X
            combined_F = np.vstack([F, np.array(offspring_F)]) if len(offspring_F) > 0 else F
            combined_F = np.array(combined_F)
            
            # Truncate to population size using non-dominated sorting and crowding distance
            if len(combined_X) > pop_size:
                fronts = fast_non_dominated_sort(combined_F)
                new_X = []
                new_F = []
                for front in fronts:
                    if len(new_X) + len(front) <= pop_size:
                        for idx in front:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                    else:
                        # Need to select from this front
                        remaining = pop_size - len(new_X)
                        if remaining > 0:
                            cd_vals = crowding_distance(combined_F, front)
                            # Sort by crowding distance descending
                            sorted_indices = sorted(range(len(front)), key=lambda i: cd_vals[i], reverse=True)
                            for i in range(remaining):
                                idx = front[sorted_indices[i]]
                                new_X.append(combined_X[idx])
                                new_F.append(combined_F[idx])
                        break
                
                X = np.array(new_X) if new_X else combined_X[:pop_size]
                F = np.array(new_F) if new_F else combined_F[:pop_size]
            else:
                X = combined_X
                F = combined_F

        return X
# EVOLVE-BLOCK-END
