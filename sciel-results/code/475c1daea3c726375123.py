import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Latin Hypercube Sampling for better initial coverage
        # We want to use a good chunk of the budget for initialization to get a diverse set.
        # Let's use 100 points if budget is large enough, otherwise half the budget.
        N_init = 100
        if self.budget < 2 * N_init:
            N_init = self.budget // 2
            if N_init < 10:
                N_init = 10 # Minimum viable population

        # Generate LHS points
        X_lhs = np.zeros((N_init, dim))
        for d in range(dim):
            # Divide range into N_init bins
            bin_width = (hi[d] - lo[d]) / N_init
            # Random permutation of bin indices
            perm = np.random.permutation(N_init)
            # Uniform random point within each bin
            X_lhs[:, d] = lo[d] + bin_width * (perm + np.random.random(N_init))
        
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = N_init

        # If we used all budget, return
        if used >= self.budget:
            return X_lhs

        # Phase 2: NSGA-II Evolution
        # Select a smaller population for evolution to allow more generations
        # We need to select the best N_evolve points from the LHS set.
        # Let's take the non-dominated ones if possible, else random.
        
        # Helper functions for NSGA-II style selection
        
        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is (N, M) array of objective values.
            """
            N = F.shape[0]
            S = [[] for _ in range(N)]  # Domination sets
            n = [0] * N  # Domination counts
            fronts = [[]]
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # Check if q dominates p
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            for p in range(N):
                if n[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i] != []:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
                
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in F.
            Returns dictionary of index -> distance.
            """
            if len(indices) == 0:
                return {}
            
            distances = {i: 0.0 for i in indices}
            
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            for m in range(n_obj):
                # Sort indices by m-th objective
                sorted_indices = sorted(indices, key=lambda k: F[k][m])
                
                # Assign infinite distance to boundary points
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max - f_min == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[k]
                    prev_idx = sorted_indices[k-1]
                    next_idx = sorted_indices[k+1]
                    
                    distances[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)
                    
            return distances

        # Select initial population for evolution from LHS
        # We want a population size that allows multiple generations.
        # Let's aim for pop_size = 20.
        pop_size = 20
        
        fronts_init = fast_non_dominated_sort(F_lhs)
        
        X = []
        F = []
        i = 0
        while len(X) < pop_size and i < len(fronts_init):
            front = fronts_init[i]
            if len(X) + len(front) <= pop_size:
                for idx in front:
                    X.append(X_lhs[idx])
                    F.append(F_lhs[idx])
                i += 1
            else:
                crowd = crowding_distance(F_lhs, front)
                sorted_front = sorted(front, key=lambda k: crowd[k], reverse=True)
                for idx in sorted_front:
                    if len(X) >= pop_size:
                        break
                    X.append(X_lhs[idx])
                    F.append(F_lhs[idx])
                break
        
        X = np.array(X)
        F = np.array(F)
        
        # Helper functions for NSGA-II style selection
        
        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is (N, M) array of objective values.
            """
            N = F.shape[0]
            S = [[] for _ in range(N)]  # Domination sets
            n = [0] * N  # Domination counts
            fronts = [[]]
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # Check if q dominates p
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            for p in range(N):
                if n[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i] != []:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
                
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in F.
            Returns dictionary of index -> distance.
            """
            if len(indices) == 0:
                return {}
            
            distances = {i: 0.0 for i in indices}
            
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            for m in range(n_obj):
                # Sort indices by m-th objective
                sorted_indices = sorted(indices, key=lambda k: F[k][m])
                
                # Assign infinite distance to boundary points
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max - f_min == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[k]
                    prev_idx = sorted_indices[k-1]
                    next_idx = sorted_indices[k+1]
                    
                    distances[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)
                    
            return distances

        def tournament_select(P, F, rank, crowd, n=2):
            """
            Binary tournament selection.
            P: list of indices in population
            F: objective values
            rank: dictionary index -> rank
            crowd: dictionary index -> crowding distance
            """
            a = np.random.choice(P)
            b = np.random.choice(P)
            
            if rank[a] < rank[b]:
                return a
            elif rank[b] < rank[a]:
                return b
            else:
                # Tie-break by crowding distance (higher is better)
                if crowd[a] > crowd[b]:
                    return a
                else:
                    return b

        def sbx_crossover(x1, x2, rate, eta):
            """
            Simulated Binary Crossover.
            """
            x3 = x1.copy()
            x4 = x2.copy()
            
            if np.random.random() > rate:
                return x3, x4
                
            for i in range(len(x1)):
                if np.random.random() < 0.5:
                    if x1[i] != x2[i]:
                        # Standard SBX implementation
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        
                        x3[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x4[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        
            # Clip to bounds
            x3 = np.clip(x3, lo, hi)
            x4 = np.clip(x4, lo, hi)
            
            return x3, x4

        def polynomial_mutate(x, rate, eta):
            """
            Polynomial Mutation.
            """
            for i in range(len(x)):
                if np.random.random() < rate:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-10)
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-10)
                    u = np.random.random()
                    if u < 0.5:
                        xy = 1.0 - delta1
                        val = (2.0 * u + (1.0 - 2.0 * u) * (xy) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 1.0 - (2.0 * (1.0 - u) + (2.0 * u - 1.0) * (xy) ** (eta + 1.0)) ** (1.0 / (eta + 1.0))
                    
                    x[i] += val * (hi[i] - lo[i])
                    
            return np.clip(x, lo, hi)

        # Main Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sorting of current population
            fronts = fast_non_dominated_sort(F)
            
            # Assign ranks
            rank = {}
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding distance for the last full front
            # In standard NSGA-II, we only need crowd distance for the front that determines the cut
            # But for selection, we need it for all. For small N, calculating for all is fine.
            crowd = {}
            for front in fronts:
                crowd.update(crowding_distance(F, front))
            
            # 2. Create mating pool
            parents = []
            for _ in range(pop_size):
                p_idx = tournament_select(range(pop_size), F, rank, crowd)
                parents.append(p_idx)
            
            # 3. Generate offspring
            offspring_X = []
            rate_c = 0.9
            rate_m = 1.0 / dim
            eta_c = 20.0
            eta_m = 20.0
            
            # Generate pop_size offspring
            # We need to check budget before evaluating.
            # Each offspring requires 1 evaluation.
            # We generate pop_size offspring.
            # If used + pop_size > budget, we can only generate (budget - used) offspring.
            # But NSGA-II assumes equal population size.
            # Let's just generate as many as we can afford, up to pop_size.
            
            n_offspring = min(pop_size, self.budget - used)
            
            for i in range(0, n_offspring, 2):
                p1 = parents[i % pop_size]
                p2 = parents[(i+1) % pop_size]
                
                x1, x2 = sbx_crossover(X[p1], X[p2], rate_c, eta_c)
                x1 = polynomial_mutate(x1, rate_m, eta_m)
                
                offspring_X.append(x1)
                
                if i + 1 < n_offspring:
                    x2 = polynomial_mutate(x2, rate_m, eta_m)
                    offspring_X.append(x2)
            
            if not offspring_X:
                break
                
            Q = np.array(offspring_X)
            F_q = np.array([problem(x) for x in Q])
            used += len(Q)
            
            # 4. Combine P and Q
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, F_q))
            
            # 5. Non-dominated sort of R
            fronts_R = fast_non_dominated_sort(R_F)
            
            # 6. Select next P
            new_X = []
            new_F = []
            new_rank = {}
            i = 0
            while len(new_X) < pop_size and i < len(fronts_R):
                front = fronts_R[i]
                if len(new_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    i += 1
                else:
                    # Need to select from this front based on crowding distance
                    crowd_R = crowding_distance(R_F, front)
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda k: crowd_R[k], reverse=True)
                    for idx in sorted_front:
                        if len(new_X) >= pop_size:
                            break
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
