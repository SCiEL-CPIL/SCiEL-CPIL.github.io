import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Set random seed for reproducibility if needed, but task says evaluator sets np.random.seed
        # We should use np.random directly or default_rng. 
        # The task says "evaluator calls np.random.seed(seed)". So we should use np.random functions.
        
        # Population size
        pop_size = min(40, budget)
        
        # Initialize population with Latin Hypercube Sampling for better spread
        X = np.zeros((pop_size, dim))
        for i in range(dim):
            # Create LHS samples
            perm = np.random.permutation(pop_size)
            X[:, i] = lo[i] + (hi[i] - lo[i]) * (perm + np.random.rand(pop_size)) / pop_size
            # Clip to ensure within bounds (LHS might slightly overshoot if not careful, but formula is safe)
            X[:, i] = np.clip(X[:, i], lo[i], hi[i])
            
        F = np.array([problem(x) for x in X])
        eval_count = pop_size
        
        def non_dominated_sort(F):
            """
            Returns list of lists of indices, sorted by non-dominated rank.
            """
            m = len(F)
            S = [[] for _ in range(m)]  # Dominated sets
            n = [0] * m  # Number of dominators
            rank = [0] * m
            fronts = [[]]
            
            for i in range(m):
                for j in range(i + 1, m):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n[i] += 1
                if n[i] == 0:
                    rank[i] = 0
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
                
            return fronts[:-1] # Remove the last empty list if any

        def crowding_distance(F, front_indices):
            """
            Calculates crowding distance for indices in a front.
            """
            if len(front_indices) <= 2:
                return np.inf * np.ones(len(front_indices))
            
            m = len(front_indices)
            cd = np.zeros(m)
            n_obj = len(F[front_indices[0]])
            
            for k in range(n_obj):
                # Sort front indices by objective k
                sorted_idx = np.argsort(F[front_indices, k])
                # Get the sorted objective values
                sorted_obj = F[front_indices, k][sorted_idx]
                
                # The best and worst get infinite distance
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                # Calculate range
                range_val = sorted_obj[-1] - sorted_obj[0]
                if range_val == 0:
                    continue
                    
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (sorted_obj[i + 1] - sorted_obj[i - 1]) / range_val
                    
            return cd

        def tournament_selection(fronts, cd_map, X):
            """
            Selects one parent using binary tournament.
            fronts: list of lists of indices
            cd_map: dict mapping index -> crowding distance
            """
            idx1 = np.random.randint(len(X))
            idx2 = np.random.randint(len(X))
            
            # Find rank
            rank1 = 0
            rank2 = 0
            for r, front in enumerate(fronts):
                if idx1 in front:
                    rank1 = r
                    break
                if idx2 in front:
                    rank2 = r
                    break
            # If not found (should not happen), assign high rank
            if rank1 == 0 and idx1 not in fronts[0]: rank1 = len(fronts)
            if rank2 == 0 and idx2 not in fronts[0]: rank2 = len(fronts)

            if rank1 < rank2:
                return X[idx1].copy()
            elif rank2 < rank1:
                return X[idx2].copy()
            else:
                # Same rank, compare crowding distance
                if cd_map[idx1] > cd_map[idx2]:
                    return X[idx1].copy()
                else:
                    return X[idx2].copy()

        while eval_count < budget:
            if time.time() > deadline:
                break
                
            # Sort current population
            fronts = non_dominated_sort(F)
            
            # Calculate crowding distance for all fronts
            cd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for idx, cd in zip(front, cd_vals):
                        cd_map[idx] = cd
            
            # Generate offspring
            n_offspring = min(pop_size, budget - eval_count)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                if eval_count + 1 > budget:
                    break
                p1 = tournament_selection(fronts, cd_map, X)
                p2 = tournament_selection(fronts, cd_map, X)
                
                # SBX Crossover
                child = np.zeros(dim)
                for j in range(dim):
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (1 + 20))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (1 + 20))
                    
                    c1 = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    c2 = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                    
                    if np.random.rand() < 0.5:
                        child[j] = c1
                    else:
                        child[j] = c2
                        
                # Polynomial Mutation
                for j in range(dim):
                    if np.random.rand() < 0.2: # Mutation rate
                        delta1 = (child[j] - lo[j]) / (hi[j] - lo[j])
                        delta2 = (hi[j] - child[j]) / (hi[j] - lo[j])
                        r = np.random.rand()
                        if r < 0.5:
                            xy = 1 + delta1
                            val = 2 * r
                        else:
                            xy = 1 + delta2
                            val = 2 * (1 - r)
                        
                        # eta_mu = 20
                        if val == 0:
                            continue
                        if np.random.rand() < 0.5:
                            mut_pow = 1.0 / (20 + 1.0)
                            val = 0.5 * (1.0 - (xy * (2.0 * val) ** (-mut_pow)))
                        else:
                            mut_pow = 1.0 / (20 + 1.0)
                            val = 0.5 * ((xy * (2.0 * (1.0 - val))) ** (-mut_pow) - 1.0)
                            
                        child[j] += val * (hi[j] - lo[j])
                        
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                eval_count += 1
                
            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Select next generation
            fronts_all = non_dominated_sort(F_all)
            
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            # Calculate crowding for all fronts in combined pop
            cd_map_all = {}
            for front in fronts_all:
                if len(front) > 0:
                    cd_vals = crowding_distance(F_all, front)
                    for idx, cd in zip(front, cd_vals):
                        cd_map_all[idx] = cd
            
            count = 0
            for front in fronts_all:
                if count + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                else:
                    # Take only part
                    needed = pop_size - count
                    # Sort by crowding distance descending
                    front_indices = np.array(front)
                    cd_vals = np.array([cd_map_all[idx] for idx in front_indices])
                    sorted_order = np.argsort(-cd_vals)
                    for i in range(needed):
                        idx = front_indices[sorted_order[i]]
                        new_X[count] = X_all[idx]
                        new_F[count] = F_all[idx]
                        count += 1
                    break
                    
            X = new_X
            F = new_F
            
        return X
