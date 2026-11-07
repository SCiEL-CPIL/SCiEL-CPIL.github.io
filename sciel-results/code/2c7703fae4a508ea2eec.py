import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in this context, though evaluator seeds np.random
        
        # NSGA-II Parameters
        pop_size = 20
        # Ensure we don't exceed budget
        # Initial pop: pop_size evals
        # Each generation: pop_size offspring evals
        # Total = pop_size * (1 + n_gen)
        # We have 200 budget. pop_size=20 -> 10 generations total (1 init + 9 gen)
        n_gen = (self.budget // pop_size) - 1
        if n_gen < 1:
            n_gen = 1
            pop_size = self.budget // 2 # Fallback if budget is small

        # Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions for NSGA-II

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j if i <= j and i != j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
            
            for i in range(n):
                if n_d[i] == 0:
                    ranks[i] = 0
                    fronts.append([i])
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts, ranks

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                # Sort front by k-th objective
                objs = F[front_indices, k]
                sorted_idx = np.argsort(objs)
                front_indices_sorted = [front_indices[i] for i in sorted_idx]
                
                # Assign inf to extremes
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                # Normalize
                range_k = objs[sorted_idx[-1]] - objs[sorted_idx[0]]
                if range_k == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx_curr = front_indices_sorted[i]
                    idx_prev = front_indices_sorted[i-1]
                    idx_next = front_indices_sorted[i+1]
                    
                    # Map back to local index in 'cd' array? 
                    # Wait, 'cd' is indexed by position in front_indices? 
                    # Let's make cd an array of length n, where n is len(front_indices)
                    # But we need to assign to the original population indices?
                    # Actually, we just need to compare within the front.
                    # Let's store cd for the front elements.
                    
                    # The formula: (f_k(i+1) - f_k(i-1)) / (f_k_max - f_k_min)
                    # We need to add this to the crowding distance of the middle point.
                    
                    # Let's re-implement to return an array of CD values corresponding to front_indices
                    pass 
            
            # Correct implementation of crowding distance
            cd = np.zeros(n)
            for k in range(self.n_obj):
                objs = F[front_indices, k]
                sorted_order = np.argsort(objs)
                # Extremes
                cd[sorted_order[0]] = np.inf
                cd[sorted_order[-1]] = np.inf
                
                min_val = objs[sorted_order[0]]
                max_val = objs[sorted_order[-1]]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx_curr = sorted_order[i]
                    idx_prev = sorted_order[i-1]
                    idx_next = sorted_order[i+1]
                    
                    # Distance in objective space
                    dist = (objs[idx_next] - objs[idx_prev]) / range_val
                    cd[idx_curr] += dist
            
            return cd

        def sbx_crossover(x1, x2, lo, hi):
            # Simulated Binary Crossover
            x1 = np.array(x1)
            x2 = np.array(x2)
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x1)
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (20 + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (20 + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    c1[i] = x2[i]
                    c2[i] = x1[i]
            
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        def polynomial_mutation(x, lo, hi):
            x = np.array(x)
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim: # Mutation rate
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (20 + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (20 + 1))
                    
                    x[i] += delta * (hi[i] - lo[i])
            
            return np.clip(x, lo, hi)

        # Main Loop
        for gen in range(n_gen):
            offspring = []
            for i in range(pop_size // 2):
                # Select 2 parents using tournament selection on ranks and crowding
                # For simplicity, pick random parents from current pop, but weighted by rank?
                # Standard NSGA-II uses binary tournament.
                
                # Parent 1
                p1_idx1, p1_idx2 = rng.integers(0, pop_size, 2)
                # We need ranks and CD for current pop
                # We calculate ranks/CD once per generation before creating offspring? 
                # Yes, select parents from current population.
                
                # Let's compute ranks and CD for current X, F
                fronts, ranks = fast_non_dominated_sort(F)
                # Calculate CD for each front
                cd = np.zeros(pop_size)
                for front in fronts:
                    cd_front = crowding_distance(F, front)
                    for idx, val in zip(front, cd_front):
                        cd[idx] = val
                
                def tournament():
                    idx1, idx2 = rng.integers(0, pop_size, 2)
                    if ranks[idx1] < ranks[idx2]:
                        return idx1
                    elif ranks[idx2] < ranks[idx1]:
                        return idx2
                    else:
                        if cd[idx1] > cd[idx2]:
                            return idx1
                        else:
                            return idx2

                p1 = tournament()
                p2 = tournament()
                
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring.append(c1)
                offspring.append(c2)
            
            offspring = np.array(offspring)
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_off])
            
            # Sort combined
            fronts, ranks = fast_non_dominated_sort(F_combined)
            cd = np.zeros(len(X_combined))
            for front in fronts:
                cd_front = crowding_distance(F_combined, front)
                for idx, val in zip(front, cd_front):
                    cd[idx] = val
            
            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend([X_combined[i] for i in front])
                    next_F.extend([F_combined[i] for i in front])
                else:
                    # Need to pick subset of this front
                    # Sort by crowding distance
                    front_cd = cd[front]
                    sorted_front = [front[i] for i in np.argsort(front_cd)[::-1]]
                    needed = pop_size - len(next_X)
                    for i in range(needed):
                        next_X.append(X_combined[sorted_front[i]])
                        next_F.append(F_combined[sorted_front[i]])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            if used >= self.budget:
                break

        return X
