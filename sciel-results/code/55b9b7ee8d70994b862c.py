import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        N = 10  # Population size
        remaining = self.budget
        
        # Initial population
        X = lo + np.random.random((N, self.dim)) * span
        F = np.array([problem(x) for x in X])
        remaining -= N
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            current_front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    # j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            I = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = I + 1
                            next_front.append(j)
                I += 1
                current_front = next_front
            return rank

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            n_obj = F.shape[1]
            dist = np.zeros(m)
            
            for k in range(n_obj):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val == min_val:
                    dist[:] = float('inf')
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / (max_val - min_val)
            return dist

        def tournament_select(P, F, rank, dist, k=2):
            indices = np.random.choice(len(P), k, replace=False)
            best_idx = indices[0]
            for i in indices[1:]:
                if rank[i] < rank[best_idx] or (rank[i] == rank[best_idx] and dist[i] > dist[best_idx]):
                    best_idx = i
            return P[best_idx]

        def sbx_crossover(x1, x2, eta=15.0):
            if np.random.random() > 0.5:
                x1, x2 = x2, x1
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Main loop
        while remaining >= N:
            if time.time() > deadline:
                break
                
            Q = []
            for _ in range(N):
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                for child in [c1, c2]:
                    mask = np.random.random(self.dim) < 1.0 / self.dim
                    child[mask] += 0.1 * span[mask] * np.random.standard_normal(np.sum(mask))
                    child[:] = np.clip(child, lo, hi)
                
                Q.append(c1)
                Q.append(c2)
            
            # We generated 2*N offspring, but we only have budget for N?
            # Wait, the loop condition `remaining >= N` implies we generate N offspring per generation.
            # My inner loop generated 2*N. I need to adjust.
            # Let's generate exactly N offspring per generation.
            
            # Re-do inner loop logic to generate exactly N
            Q = []
            for _ in range(N // 2):
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2)
                
                for child in [c1, c2]:
                    mask = np.random.random(self.dim) < 1.0 / self.dim
                    child[mask] += 0.1 * span[mask] * np.random.standard_normal(np.sum(mask))
                    child[:] = np.clip(child, lo, hi)
                
                Q.append(c1)
                Q.append(c2)
            
            # If N is odd, we might have an issue. N=10 is even.
            # Evaluate Q
            F_Q = np.array([problem(x) for x in Q])
            remaining -= N
            
            # Combine
            R = np.vstack([X, Q])
            F_R = np.vstack([F, F_Q])
            
            # Sort and Select
            rank_R = fast_non_dominated_sort(F_R)
            dist_R = np.zeros(len(R))
            
            # Calculate crowding distance for the last selected front?
            # Standard NSGA-II:
            # 1. Sort by rank.
            # 2. Add fronts to new pop until full.
            # 3. If the next front exceeds capacity, sort that front by crowding and take the rest.
            
            new_X = []
            new_F = []
            current_rank = 0
            while len(new_X) < N:
                front_indices = [i for i in range(len(R)) if rank_R[i] == current_rank]
                if len(new_X) + len(front_indices) <= N:
                    for i in front_indices:
                        new_X.append(R[i])
                        new_F.append(F_R[i])
                else:
                    # Need to select only some from this front
                    dist_front = crowding_distance(F_R, front_indices)
                    sorted_front = np.argsort(-dist_front)
                    needed = N - len(new_X)
                    for i in sorted_front[:needed]:
                        idx = front_indices[i]
                        new_X.append(R[idx])
                        new_F.append(F_R[idx])
                    break
                current_rank += 1
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Recalculate rank and dist for next tournament
            rank = fast_non_dominated_sort(F)
            dist = np.zeros(N)
            # Simplified: just use rank for now, or calculate full dist
            # For efficiency, we can calculate dist on the whole population
            dist = crowding_distance(F, list(range(N)))

        return X
