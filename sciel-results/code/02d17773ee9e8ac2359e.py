import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 295.0  # 5s margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        N = 20  # Population size
        
        # 1. Initialization
        X = np.random.uniform(lo, hi, (N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Dominance matrix
        # A[i] dominates A[j] if all(A[i] <= A[j]) and any(A[i] < A[j])
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_pop):
            """
            Returns ranks (list of indices for each front) and crowding distances.
            F_pop: (N, M) array
            """
            N_pop = len(F_pop)
            ranks = [0] * N_pop
            domination_count = [0] * N_pop
            dominated_set = [[] for _ in range(N_pop)]
            
            # Calculate dominance relationships
            for i in range(N_pop):
                for j in range(i + 1, N_pop):
                    if dominates(F_pop[i], F_pop[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif dominates(F_pop[j], F_pop[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            fronts = []
            current_front = []
            for i in range(N_pop):
                if domination_count[i] == 0:
                    ranks[i] = 0
                    current_front.append(i)
            
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            
            # Crowding Distance
            crowding = [0.0] * N_pop
            for f_idx, front in enumerate(fronts):
                if len(front) <= 2:
                    for i in front:
                        crowding[i] = float('inf')
                    continue
                
                for m in range(n_obj):
                    # Sort front by objective m
                    sorted_indices = sorted(front, key=lambda idx: F_pop[idx][m])
                    crowding[sorted_indices[0]] = float('inf')
                    crowding[sorted_indices[-1]] = float('inf')
                    
                    range_val = F_pop[sorted_indices[-1]][m] - F_pop[sorted_indices[0]][m]
                    if range_val == 0:
                        continue
                        
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        prev_idx = sorted_indices[i - 1]
                        next_idx = sorted_indices[i + 1]
                        crowding[idx] += (F_pop[next_idx][m] - F_pop[prev_idx][m]) / range_val
            
            return ranks, fronts, crowding

        # Main Loop
        # We have 180 evaluations left. 9 generations of 20.
        generations = (self.budget - used) // N
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Rank current population
            ranks, fronts, crowding = non_dominated_sort(F)
            
            # Create selection pool for tournament
            # We need to select parents from P
            # Tournament size 2
            # Key: (rank, -crowding) -> lower rank better, higher crowding better
            
            def tournament_select(pop_idx_list, pop_F, pop_ranks, pop_crowding):
                i1 = np.random.randint(0, N)
                i2 = np.random.randint(0, N)
                
                # Compare
                # Lower rank is better
                if pop_ranks[i1] < pop_ranks[i2]:
                    return i1
                elif pop_ranks[i2] < pop_ranks[i1]:
                    return i2
                else:
                    # Same rank, higher crowding is better
                    if pop_crowding[i1] > pop_crowding[i2]:
                        return i1
                    else:
                        return i2

            Q_X = np.zeros((N, dim))
            Q_F = np.zeros((N, n_obj))
            
            for i in range(N):
                # Select 2 parents
                p1 = tournament_select(None, F, ranks, crowding)
                p2 = tournament_select(None, F, ranks, crowding)
                
                # SBX Crossover
                # eta = 20
                # For each dimension
                child = np.zeros(dim)
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        # Crossover
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 21)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        
                        x1 = X[p1][d]
                        x2 = X[p2][d]
                        
                        if abs(x1 - x2) < 1e-14:
                            child[d] = x1
                        else:
                            c1 = 0.5 * (1 + beta) * x1 + 0.5 * (1 - beta) * x2
                            child[d] = c1
                    else:
                        child[d] = X[p1][d]
                
                # Polynomial Mutation
                # eta = 20
                for d in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 21) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 21)
                        
                        range_val = hi[d] - lo[d]
                        child[d] += delta * range_val
                        
                        # Clip
                        child[d] = np.clip(child[d], lo[d], hi[d])
                
                Q_X[i] = child
                Q_F[i] = problem(child)
            
            used += N
            
            # Combine P and Q
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Rank R
            ranks_R, fronts_R, crowding_R = non_dominated_sort(R_F)
            
            # Select new P from R
            new_P_indices = []
            for front in fronts_R:
                if len(new_P_indices) + len(front) <= N:
                    new_P_indices.extend(front)
                else:
                    # Need to select from this front based on crowding
                    remaining = N - len(new_P_indices)
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda idx: crowding_R[idx], reverse=True)
                    new_P_indices.extend(front_sorted[:remaining])
                    break
            
            X = R_X[new_P_indices]
            F = R_F[new_P_indices]
            
        return X
