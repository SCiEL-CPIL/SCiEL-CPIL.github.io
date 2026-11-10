import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Hyperparameters
        pop_size = 20
        crossover_prob = 0.9
        mutation_prob = 1.0 / dim
        eta_c = 15.0
        eta_m = 20.0
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominates[i] += 1
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
            
            for i in range(n):
                if n_dominates[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k] != []:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            return fronts[:-1]

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            dist = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(n_obj):
                # Sort by k-th objective
                sorted_indices = np.argsort(F_sub[:, k])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_range = F_sub[sorted_indices[-1], k] - F_sub[sorted_indices[0], k]
                if f_range == 0:
                    continue
                
                for i in range(1, m - 1):
                    dist[sorted_indices[i]] += (F_sub[sorted_indices[i+1], k] - F_sub[sorted_indices[i-1], k]) / f_range
            
            return dist

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, lo, hi):
            if np.random.rand() > crossover_prob:
                return x1.copy(), x2.copy()
            
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(dim):
                if np.random.rand() < 0.5:
                    x1_i, x2_i = x1[i], x2[i]
                else:
                    x1_i, x2_i = x2[i], x1[i]
                
                if x1_i == x2_i:
                    child1[i] = x1_i
                    child2[i] = x2_i
                    continue
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                
                c1 = 0.5 * ((1.0 + beta) * x1_i + (1.0 - beta) * x2_i)
                c2 = 0.5 * ((1.0 - beta) * x1_i + (1.0 + beta) * x2_i)
                
                # Clip to bounds
                child1[i] = np.clip(c1, lo[i], hi[i])
                child2[i] = np.clip(c2, lo[i], hi[i])
            
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, lo, hi):
            for i in range(dim):
                if np.random.rand() < mutation_prob:
                    u = np.random.rand()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    
                    x[i] += delta_q * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Initialization
        rng = np.random.default_rng(0)
        # Use rng for reproducibility consistent with seed, but problem is seeded externally?
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # So I should use np.random, not default_rng, to respect the external seed?
        # The prompt says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)..."
        # So I should use np.random functions.
        
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            Q_X = []
            for _ in range(pop_size // 2):
                # Pick two parents
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                
                c1, c2 = sbx_crossover(X[idx1], X[idx2], lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                Q_X.append(c1)
                Q_X.append(c2)
            
            Q_F = np.array([problem(x) for x in Q_X])
            used += pop_size
            
            # Combine
            R_X = np.vstack((X, Q_X))
            R_F = np.vstack((F, Q_F))
            
            # Sort and Select
            fronts = non_dominated_sort(R_F)
            
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need crowding distance
                    dist = crowding_distance(R_F, front)
                    # Sort front by distance descending
                    sorted_front = [front[i] for i in np.argsort(dist)[::-1]]
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
