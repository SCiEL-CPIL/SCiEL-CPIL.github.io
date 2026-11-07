import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size and generations to fit budget
        # Budget = 200. Dim = 12.
        # Let's use P=20, G=9. 20 + 9*20 = 200.
        P = 20
        G = (self.budget - P) // P
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((P, self.dim))
        F = np.array([problem(x) for x in X])
        used = P
        
        # NSGA-II Loop
        for g in range(G):
            if time.time() > deadline:
                break
            
            # Combine parents and offspring? No, standard NSGA-II:
            # 1. Tournament select parents
            # 2. Crossover/Mutate
            # 3. Evaluate offspring
            # 4. Combine P + Offspring
            # 5. Non-dominated sort
            # 6. Select P best for next gen
            
            # For simplicity and speed with small P, we can just generate new population
            # and select from union of old and new.
            
            # Generate offspring
            X_off = np.zeros_like(X)
            for i in range(P):
                # Tournament selection
                idx1 = np.random.choice(P, 2, replace=False)
                # Simple dominance check for tournament
                # If idx1[0] dominates idx1[1], pick 0, else 1.
                # For 3 objectives, this is fast.
                if self._dominates(F[idx1[0]], F[idx1[1]]):
                    parent1_idx = idx1[0]
                else:
                    parent1_idx = idx1[1]
                
                idx2 = np.random.choice(P, 2, replace=False)
                if self._dominates(F[idx2[0]], F[idx2[1]]):
                    parent2_idx = idx2[0]
                else:
                    parent2_idx = idx2[1]
                
                # SBX Crossover
                # eta = 20
                beta = 1.0
                if np.random.rand() < 0.5:
                    beta = 1.0 / (2.0 * (1.0 - np.random.rand()))
                else:
                    beta = 1.0 / (2.0 * np.random.rand())
                
                x1 = X[parent1_idx]
                x2 = X[parent2_idx]
                
                child1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                child2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                
                # Polynomial Mutation
                # eta = 20
                for c in [child1, child2]:
                    for j in range(self.dim):
                        if np.random.rand() < 0.9: # Mutation rate
                            delta1 = (c[j] - lo[j]) / (hi[j] - lo[j])
                            delta2 = (hi[j] - c[j]) / (hi[j] - lo[j])
                            if np.random.rand() < 0.5:
                                xy = 1.0 - (1.0 - delta1) ** (eta + 1)
                            else:
                                xy = 1.0 - (1.0 - delta2) ** (eta + 1)
                            c[j] = c[j] + (hi[j] - lo[j]) * xy
                            c[j] = np.clip(c[j], lo[j], hi[j])
                
                X_off[i] = child1 # We only need P offspring, so we discard child2 or use it?
                # Standard EA: 1 offspring per pair.
                # To get P offspring from P pairs, we use one child per pair.
                # Let's use child1.
            
            F_off = np.array([problem(x) for x in X_off])
            used += P
            
            # Combine
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sort
            ranks = self._fast_non_dominated_sort(F_all)
            
            # Sort by rank, then by crowding distance
            # We need to select P individuals.
            # 1. Sort by rank
            # 2. For the last rank that doesn't fit, sort by crowding distance
            
            # Simple approach: 
            # 1. Get non-dominated front (rank 0)
            # 2. If len < P, add next front, etc.
            
            selected_idx = []
            current_front_size = 0
            front_idx = 0
            while current_front_size < P and front_idx < len(np.unique(ranks)):
                mask = (ranks == front_idx)
                front_indices = np.where(mask)[0]
                if current_front_size + len(front_indices) <= P:
                    selected_idx.extend(front_indices)
                    current_front_size += len(front_indices)
                else:
                    # Need to select only some from this front based on crowding
                    # Calculate crowding distance for this front
                    front_F = F_all[front_indices]
                    cd = self._crowding_distance(front_F)
                    # Sort by cd descending
                    sorted_order = np.argsort(-cd)
                    needed = P - current_front_size
                    selected_idx.extend(front_indices[sorted_order[:needed]])
                    current_front_size += needed
                front_idx += 1
            
            X = X_all[selected_idx]
            F = F_all[selected_idx]
            
        return X

    def _dominates(self, a, b):
        # a dominates b if a <= b in all objectives and a < b in at least one
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_count[i] += 1
                elif self._dominates(F[j], F[i]):
                    n_count[j] += 1
            
            if n_count[i] == 0:
                ranks[i] = 0
        
        I = [i for i in range(n) if ranks[i] == 0]
        k = 1
        while len(I) > 0:
            Q = []
            for i in I:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        ranks[j] = k
                        Q.append(j)
            I = Q
            k += 1
        return ranks

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(self.n_obj):
            # Sort by objective m
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            range_val = F[order[-1], m] - F[order[0], m]
            if range_val == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / range_val
        return cd
