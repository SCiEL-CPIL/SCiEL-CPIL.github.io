import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # If budget is small, use a smaller pop. 
        # 200 budget -> pop 20, 9 generations (20 init + 9*20 offspring = 200)
        # Let's try pop_size = 20
        pop_size = min(20, budget)
        if pop_size < 4:
            pop_size = 4 # Minimum viable pop
            
        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sorting and crowding distance
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        S[j].add(i)
                
                if n_dom[i] == 0:
                    ranks[i] = 0
                    fronts.append([i])
                    
            k = 0
            while fronts[k] is not None:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            # Crowding distance
            cd = np.zeros(n)
            for front in fronts:
                if len(front) == 1:
                    cd[front[0]] = np.inf
                    continue
                for m in range(n_obj):
                    f_min = F[front, m].min()
                    f_max = F[front, m].max()
                    if f_max == f_min:
                        continue
                    # Sort by objective m
                    sorted_idx = sorted(front, key=lambda i: F[i, m])
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        idx_curr = sorted_idx[i]
                        idx_prev = sorted_idx[i-1]
                        idx_next = sorted_idx[i+1]
                        cd[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / (f_max - f_min)
                        
            return ranks, cd

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))
            
            for i in range(pop_size):
                # Selection
                p1 = self._tournament(X, F, 2, rng)
                p2 = self._tournament(X, F, 2, rng)
                
                # Crossover (SBX-like, simplified)
                if rng.random() < 0.9:
                    child = 0.5 * (p1 + p2) + 0.1 * (hi - lo) * rng.standard_normal(dim)
                else:
                    child = p1.copy()
                
                # Mutation
                # Gaussian mutation
                if rng.random() < 0.9:
                    child += 0.1 * (hi - lo) * rng.standard_normal(dim)
                
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
            
            if used >= budget:
                break
            
            # Combine and select
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            ranks, cd = fast_non_dominated_sort(F_comb)
            
            # Select next population
            next_X = []
            next_F = []
            sorted_indices = np.argsort(ranks)
            # We need to pick pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            # Create a list of (rank, -cd, index)
            selection_list = []
            for i in range(len(F_comb)):
                selection_list.append((ranks[i], -cd[i], i))
            selection_list.sort()
            
            for _, _, idx in selection_list[:pop_size]:
                next_X.append(X_comb[idx])
                next_F.append(F_comb[idx])
                
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X

    def _tournament(self, X, F, k, rng):
        # Simple tournament based on non-dominated sorting is complex to implement on the fly
        # Here we use a simplified tournament: pick k random, return the best
        # "Best" is defined by dominance, but calculating dominance in tournament is O(k^2 * n_obj)
        # For small k, it's fine.
        indices = rng.integers(0, len(X), size=k)
        best_idx = indices[0]
        for i in range(1, k):
            idx = indices[i]
            # If best_idx dominates idx, keep best_idx
            # If idx dominates best_idx, update best_idx
            # If incomparable, keep best_idx (or random, but let's keep best)
            if np.all(F[idx] <= F[best_idx]) and np.any(F[idx] < F[best_idx]):
                best_idx = idx
        return X[best_idx]
