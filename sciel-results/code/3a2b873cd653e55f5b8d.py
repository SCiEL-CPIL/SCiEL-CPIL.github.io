import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of lists of indices, where each inner list is a non-dominated front.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # Dominance sets
        n_dominating = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dominating[p] += 1
            if n_dominating[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dominating[q] -= 1
                    if n_dominating[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for points in a front.
        """
        if len(indices) <= 2:
            return np.full(len(indices), float('inf'))
        
        distances = np.zeros(len(indices))
        m = self.n_obj
        for k in range(m):
            Fk = F[indices, k]
            min_k = np.min(Fk)
            max_k = np.max(Fk)
            if max_k == min_k:
                continue
            # Sort indices by objective k
            sorted_indices = indices[np.argsort(Fk)]
            # Assign inf to extreme points
            # Find positions in the original indices array
            pos_first = np.where(indices == sorted_indices[0])[0][0]
            pos_last = np.where(indices == sorted_indices[-1])[0][0]
            distances[pos_first] = float('inf')
            distances[pos_last] = float('inf')
            
            for i in range(1, len(sorted_indices) - 1):
                idx_in_front = np.where(indices == sorted_indices[i])[0][0]
                prev_val = F[sorted_indices[i-1], k]
                next_val = F[sorted_indices[i+1], k]
                distances[idx_in_front] += (next_val - prev_val) / (max_k - min_k)
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Population size
        N = 50
        
        # Phase 1: LHS Initialization
        X = np.zeros((N, dim))
        for d in range(dim):
            perm = np.random.permutation(N)
            X[:, d] = (perm + np.random.rand(N)) / N
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Phase 2: Evolutionary Loop
        # We have budget - N evaluations left.
        # We will generate N offspring per generation.
        # Max generations = (budget - N) // N
        
        max_gen = (budget - N) // N
        
        for gen in range(max_gen):
            if time.time() > deadline:
                break
            
            # Generate Offspring using Differential Evolution
            X_off = np.zeros_like(X)
            for i in range(N):
                # Select 3 distinct parents
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                # DE/3/1 mutation
                F_mutation = 0.8
                v = X[r1] + F_mutation * (X[r2] - X[r3])
                
                # Binomial Crossover
                CR = 0.5
                j_rand = np.random.randint(dim)
                u = np.random.random(dim) < CR
                u[j_rand] = True # Ensure at least one gene is from mutant
                
                u_child = np.where(u, v, X[i])
                
                # Clip to bounds
                X_off[i] = np.clip(u_child, lo, hi)
            
            # Evaluate Offspring
            F_off = np.array([problem(x) for x in X_off])
            evals += N
            
            # Combine Parent and Offspring
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])
            
            # Selection: NSGA-II
            fronts = self._fast_non_dominated_sort(F_combined)
            
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = N - len(selected_indices)
                    cd = self._crowding_distance(F_combined, front)
                    # Sort front by crowding distance descending
                    front_indices = np.array(front)
                    sorted_front = front_indices[np.argsort(cd)[::-1]]
                    selected_indices.extend(sorted_front[:remaining])
                    break
            
            # Update Population
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X
