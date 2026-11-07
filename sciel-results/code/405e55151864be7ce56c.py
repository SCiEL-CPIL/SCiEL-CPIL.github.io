import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, keep safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # Small budget, so small pop. 20 is a good balance.
        pop_size = 20
        
        # Initialize population
        # Use random uniform sampling
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        
        evaluations = pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            front = []
            k = 0
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            while current_front:
                front.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            return rank, front
        
        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.ones(len(front_indices)) * np.inf
            
            m = len(front_indices)
            cd = np.zeros(m)
            F_sub = F[front_indices]
            
            for k in range(n_obj):
                # Sort by objective k
                sorted_idx = np.argsort(F_sub[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F_sub[sorted_idx[0], k]
                f_max = F_sub[sorted_idx[-1], k]
                if f_max - f_min == 0:
                    continue
                for i in range(1, m - 1):
                    idx = sorted_idx[i]
                    cd[idx] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / (f_max - f_min)
            
            return cd
        
        # Main Loop
        while evaluations < budget:
            if time.time() > deadline - 5:
                break
            
            # Generate offspring
            # Use SBX and Polynomial Mutation
            offspring = np.zeros_like(X)
            for i in range(pop_size):
                # Select parents using tournament selection
                # Pick 2 random parents, choose the one with better rank/crowding
                # Since we need to be fast, simple tournament on rank+cd is good.
                # We need ranks and cds from the previous generation.
                # For the first iteration, we don't have them, so we just pick random or use the initial sort.
                
                # To keep it simple and fast, we'll compute ranks/cds once per generation.
                # But here we are generating offspring one by one.
                # Better: Generate whole offspring population at once?
                # With budget 200 and pop 20, we have ~10 generations.
                # Generating 20 offspring per generation is fine.
                
                pass # Placeholder for logic below

            # Let's restructure to generate a full offspring population per generation step
            # This is cleaner.
            
            # We need ranks and crowding distances of current population X
            rank, fronts = non_dominated_sort(F)
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd_sub = crowding_distance(F, front)
                    for idx, val in zip(front, cd_sub):
                        cd[idx] = val
            
            # Generate offspring
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            for i in range(pop_size):
                # Tournament selection for parent 1
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                # Compare (rank, -cd)
                if rank[p1_idx] < rank[p2_idx] or (rank[p1_idx] == rank[p2_idx] and cd[p1_idx] > cd[p2_idx]):
                    p1 = X[p1_idx]
                else:
                    p1 = X[p2_idx]
                
                # Tournament selection for parent 2
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                if rank[p3_idx] < rank[p4_idx] or (rank[p3_idx] == rank[p4_idx] and cd[p3_idx] > cd[p4_idx]):
                    p2 = X[p3_idx]
                else:
                    p2 = X[p4_idx]
                
                # SBX Crossover
                # eta_c = 20
                u = np.random.random(dim)
                beta = np.zeros(dim)
                for d in range(dim):
                    if u[d] <= 0.5:
                        beta[d] = (2 * u[d]) ** (1 / 21)
                    else:
                        beta[d] = (2 * (1 - u[d])) ** (1 / 21)
                
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                
                # Mutation (Polynomial)
                # eta_m = 20
                # Apply to c1
                x = c1
                for d in range(dim):
                    u = np.random.random()
                    delta1 = (2 * u) ** (1 / 21) - 1
                    delta2 = 1 - (2 * (1 - u)) ** (1 / 21)
                    delta = delta1 if np.random.random() < 0.5 else delta2
                    x[d] += delta * (hi[d] - lo[d])
                    x[d] = np.clip(x[d], lo[d], hi[d])
                
                new_X[i] = x
                new_F[i] = problem(x)
                evaluations += 1
                
                if evaluations >= budget:
                    break
                if time.time() > deadline - 5:
                    break
            
            if evaluations >= budget:
                break
                
            # Combine
            X_combined = np.vstack([X, new_X])
            F_combined = np.vstack([F, new_F])
            
            # Non-dominated sort combined
            rank_comb, fronts_comb = non_dominated_sort(F_combined)
            
            # Select next population
            next_X = np.zeros_like(X)
            next_F = np.zeros_like(F)
            count = 0
            for front in fronts_comb:
                if count + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        next_X[count] = X_combined[idx]
                        next_F[count] = F_combined[idx]
                        count += 1
                else:
                    # Need to select from this front
                    # Calculate crowding distance for this front
                    cd_front = crowding_distance(F_combined, front)
                    # Sort by cd descending
                    sorted_front = np.argsort(-cd_front)
                    needed = pop_size - count
                    for j in range(needed):
                        idx = front[sorted_front[j]]
                        next_X[count] = X_combined[idx]
                        next_F[count] = F_combined[idx]
                        count += 1
                    break
            
            X = next_X
            F = next_F

        return X
