import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Initialize population
        pop_size = 50
        X = rng.uniform(lo, hi, size=(pop_size, d))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        n_evals = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            ranks = np.zeros(n, dtype=int)
            domination_count = np.zeros(n, dtype=int)
            dominated_by = [set() for _ in range(n)]
            
            # Compute domination relationships
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_by[i].add(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_by[j].add(i)
                        domination_count[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                
            # Assign ranks
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                    
            return fronts, ranks

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), float('inf'))
            
            m = len(indices)
            cd = np.zeros(m)
            n_obj = F.shape[1]
            
            for k in range(n_obj):
                obj_vals = F[indices, k]
                order = np.argsort(obj_vals)
                min_val = obj_vals[order[0]]
                max_val = obj_vals[order[-1]]
                
                if max_val == min_val:
                    continue
                    
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                
                for j in range(1, m-1):
                    cd[order[j]] += (obj_vals[order[j+1]] - obj_vals[order[j-1]]) / (max_val - min_val)
                    
            return cd

        # Main loop
        while n_evals < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts, ranks = non_dominated_sort(F)
            
            # Calculate crowding distances for each front
            crowding = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, c in zip(front, cd):
                    crowding[i] = c
            
            # Create offspring
            offspring = []
            while len(offspring) < pop_size:
                if n_evals + len(offspring) >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Binary Tournament Selection
                def tournament():
                    i1 = rng.integers(len(F))
                    i2 = rng.integers(len(F))
                    
                    # Compare ranks
                    if ranks[i1] < ranks[i2]:
                        return i1
                    elif ranks[i2] < ranks[i1]:
                        return i2
                    else:
                        # Compare crowding distance
                        if crowding[i1] > crowding[i2]:
                            return i1
                        else:
                            return i2
                
                p1 = tournament()
                p2 = tournament()
                
                # SBX Crossover
                child = np.zeros(d)
                if rng.random() < 0.9:
                    for j in range(d):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u < 0.5:
                                beta_q = (2 * u) ** (1 / 21.0)
                            else:
                                beta_q = (2 * (1 - u)) ** (1 / 21.0)
                            
                            if rng.random() < 0.5:
                                child[j] = 0.5 * ((1 + beta_q) * X[p1, j] + (1 - beta_q) * X[p2, j])
                            else:
                                child[j] = 0.5 * ((1 + beta_q) * X[p2, j] + (1 - beta_q) * X[p1, j])
                        else:
                            child[j] = X[p1, j]
                else:
                    child = X[p1].copy()
                
                # Polynomial Mutation
                if rng.random() < 0.9:
                    for j in range(d):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1 / 21.0) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1 / 21.0)
                            child[j] = child[j] + delta_q * (hi[j] - lo[j])
                
                child = np.clip(child, lo, hi)
                offspring.append(child)
            
            if not offspring:
                break
                
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring])
            n_evals += len(offspring)
            
            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring)])
            F_comb = np.vstack([F, F_offspring])
            
            # Selection
            fronts_comb, ranks_comb = non_dominated_sort(F_comb)
            
            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd_front)
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
