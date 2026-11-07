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
        
        # Population size
        N = 20
        
        # Initial population
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions for NSGA-II
        
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[j] += 1
                
                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            return fronts, rank

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(indices)
            dist = {i: 0.0 for i in indices}
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[indices[sorted_idx[0]]] = float('inf')
                dist[indices[sorted_idx[-1]]] = float('inf')
                
                f_min = F_sub[sorted_idx[0], k]
                f_max = F_sub[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                
                for i in range(1, m - 1):
                    idx = indices[sorted_idx[i]]
                    dist[idx] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / (f_max - f_min)
            
            return dist

        def tournament_select(pop_F, pop_rank, pop_dist, k=2):
            best = None
            best_val = None
            for _ in range(k):
                idx = np.random.randint(len(pop_F))
                val = (pop_rank[idx], -pop_dist[idx])
                if best is None or val < best_val:
                    best = idx
                    best_val = val
            return best

        # Main loop
        while used < self.budget and time.time() < deadline:
            # Create offspring
            Q_X = []
            Q_F = []
            
            # Generate N offspring
            while len(Q_X) < N:
                # Select parents
                p1_idx = tournament_select(F, rank, dist)
                p2_idx = tournament_select(F, rank, dist)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # SBX Crossover
                if np.random.rand() < 0.9: # Crossover probability
                    eta_c = 20.0
                    child1 = np.zeros_like(x1)
                    child2 = np.zeros_like(x1)
                    for j in range(dim):
                        if np.random.rand() < 0.5:
                            beta = 1.0
                            if x1[j] == x2[j]:
                                beta = 1.0
                            else:
                                if np.random.rand() < 0.5:
                                    beta = (2 * np.random.rand()) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta_c + 1))
                            
                            c1 = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                            c2 = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                            child1[j] = c1
                            child2[j] = c2
                        else:
                            child1[j] = x1[j]
                            child2[j] = x2[j]
                    
                    # Polynomial Mutation
                    eta_m = 20.0
                    for j in range(dim):
                        if np.random.rand() < 1.0 / dim:
                            delta_1 = (2 * np.random.rand()) ** (1.0 / (eta_m + 1)) - 1.0
                            delta_2 = 1.0 - (2 * (1.0 - np.random.rand())) ** (1.0 / (eta_m + 1))
                            child1[j] += (hi[j] - lo[j]) * delta_1
                            child2[j] += (hi[j] - lo[j]) * delta_2
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    Q_X.append(child1)
                    if len(Q_X) < N:
                        Q_X.append(child2)
                else:
                    Q_X.append(x1.copy())
                    if len(Q_X) < N:
                        Q_X.append(x2.copy())
            
            Q_X = np.array(Q_X[:N])
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # Combine P and Q
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Non-dominated sort
            fronts, rank = non_dominated_sort(R_F)
            
            # Crowding distance
            dist = {i: 0.0 for i in range(len(R_F))}
            for front in fronts:
                cd = crowding_distance(R_F, front)
                for i in front:
                    dist[i] = cd[i]
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for i in front:
                        new_X.append(R_X[i])
                        new_F.append(R_F[i])
                else:
                    # Sort front by crowding distance
                    sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                    needed = N - len(new_X)
                    for i in sorted_front[:needed]:
                        new_X.append(R_X[i])
                        new_F.append(R_F[i])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Update rank and dist for next iteration selection
            # We need to re-calculate rank and dist for X only? 
            # Actually, tournament selection uses rank and dist. 
            # The rank/dist calculated on R_F are valid for the selected X if we map them correctly.
            # But it's easier to just re-calculate for the new population X.
            # However, re-calculating is O(N^2). N=20, so it's cheap.
            fronts_X, rank = non_dominated_sort(F)
            dist = {i: 0.0 for i in range(N)}
            for front in fronts_X:
                cd = crowding_distance(F, front)
                for i in front:
                    dist[i] = cd[i]

        return X
