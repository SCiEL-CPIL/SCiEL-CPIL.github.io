import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 290  # Safety margin (limit is 300s)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size: N=20 allows for 10 generations (20 init + 9*20 offspring = 200)
        # Or N=10 for 20 generations. N=20 is generally better for hypervolume.
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)
            
        # Initialize population with Latin Hypercube Sampling (LHS) for better initial coverage
        X = np.zeros((N, self.dim))
        for i in range(N):
            for j in range(self.dim):
                # LHS: divide each dimension into N equal segments, pick one random point in each
                # This ensures a more uniform spread than simple random sampling
                X[i, j] = lo[j] + (hi[j] - lo[j]) * ((i + rng.random()) / N)
        
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = F.shape[0]
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of solutions that dominate i
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                        
            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                rank += 1
                current_front = next_front
            return ranks
        
        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            n = F.shape[0]
            distances = np.zeros(n)
            if n <= 2:
                return distances
                
            # Group by rank
            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    continue
                    
                for m in range(self.n_obj):
                    # Sort by m-th objective
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    distances[sorted_idx[0]] = np.inf
                    distances[sorted_idx[-1]] = np.inf
                    
                    obj_min = F[sorted_idx[0], m]
                    obj_max = F[sorted_idx[-1], m]
                    obj_range = obj_max - obj_min
                    if obj_range == 0:
                        continue
                        
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i - 1]
                        next_idx = sorted_idx[i + 1]
                        distances[idx] += (F[next_idx, m] - F[prev_idx, m]) / obj_range
                        
            return distances

        # Helper: SBX Crossover
        def sbx_crossover(p1, p2, lo, hi, eta, rng):
            c1 = p1.copy()
            c2 = p2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (2 * u) ** (1 / (eta + 1))
                        beta = (2 * (1 - u)) ** (1 / (eta + 1))
                    else:
                        alpha = (2 * (1 - u)) ** (1 / (eta + 1))
                        beta = (2 * u) ** (1 / (eta + 1))
                    
                    # c1 = 0.5 * ((1 + alpha) * p1 + (1 - alpha) * p2)
                    # c2 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c1[i] = 0.5 * ((1 + alpha) * p1[i] + (1 - alpha) * p2[i])
                    c2[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        # Helper: Polynomial Mutation (more standard than Gaussian for bounded spaces)
        def polynomial_mutation(x, lo, hi, eta, rng):
            x_new = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    # Scale delta to the range
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        # Main Loop
        eta_crossover = 20  # Higher eta = less aggressive crossover
        prob_crossover = 0.9
        eta_mutation = 20   # Higher eta = less aggressive mutation
        prob_mutation = 1.0 / self.dim  # Standard mutation rate

        while evals < self.budget:
            if time.time() - start_time > timeout:
                break
            
            # Select parents
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)
            
            offspring_X = []
            
            num_pairs = N // 2
            if N % 2 != 0:
                num_pairs += 1
                
            for _ in range(num_pairs):
                # Tournament selection for 2 parents
                parents = []
                for _ in range(2):
                    i, j = rng.integers(0, N, size=2)
                    if ranks[i] != ranks[j]:
                        p = i if ranks[i] < ranks[j] else j
                    else:
                        p = i if cd[i] > cd[j] else j
                    parents.append(X[p])
                
                if rng.random() < prob_crossover:
                    c1, c2 = sbx_crossover(parents[0], parents[1], lo, hi, eta_crossover, rng)
                    c1 = polynomial_mutation(c1, lo, hi, eta_mutation, rng)
                    c2 = polynomial_mutation(c2, lo, hi, eta_mutation, rng)
                    offspring_X.append(c1)
                    offspring_X.append(c2)
                else:
                    c1 = polynomial_mutation(parents[0].copy(), lo, hi, eta_mutation, rng)
                    c2 = polynomial_mutation(parents[1].copy(), lo, hi, eta_mutation, rng)
                    offspring_X.append(c1)
                    offspring_X.append(c2)
            
            offspring_X = np.array(offspring_X[:N])
            
            if evals + N > self.budget:
                num_to_eval = self.budget - evals
                offspring_X = offspring_X[:num_to_eval]
            
            if len(offspring_X) == 0:
                break
                
            if time.time() - start_time > timeout:
                break
                
            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += len(offspring_F)
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Sort and Select
            ranks_comb = non_dominated_sort(F_comb)
            cd_comb = crowding_distance(F_comb, ranks_comb)
            
            selection_keys = np.stack([ranks_comb, -cd_comb], axis=1)
            order = np.lexsort((selection_keys[:, 1], selection_keys[:, 0]))
            
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]
            
        return X
