import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 15 
        if budget < 2 * N:
            N = max(2, budget // 2)
            
        # Initialize
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II Parameters
        eta_c = 15.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / dim
        
        def sbx_crossover(x1, x2):
            if rng.random() > p_c:
                return x1.copy(), x2.copy()
            
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(dim):
                if rng.random() < 0.5:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    if x1[i] == x2[i]:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        beta = 1.0 / (2.0 * rng.random() + 1.0)
                        if beta > 1.0:
                            beta = 1.0 / (beta + 1e-9)
                        alpha = 1.0 + 2.0 * min((x1[i]-lo[i])/(hi[i]-lo[i]+1e-9), (hi[i]-x2[i])/(hi[i]-lo[i]+1e-9))
                        beta_q1 = (1.0 + (1.0 + alpha) * (beta - 1.0)) ** (1.0 / (eta_c + 1.0))
                        beta_q2 = (1.0 + (1.0 + (2.0 - alpha)) * (1.0/beta - 1.0)) ** (1.0 / (eta_c + 1.0))
                        
                        c1 = 0.5 * (x1[i] + x2[i]) - 0.5 * (x1[i] - x2[i]) * beta_q1
                        c2 = 0.5 * (x1[i] + x2[i]) + 0.5 * (x1[i] - x2[i]) * beta_q2
                        
                        c1 = np.clip(c1, lo[i], hi[i])
                        c2 = np.clip(c2, lo[i], hi[i])
                        
                        child1[i] = c1
                        child2[i] = c2
            return child1, child2

        def polynomial_mutation(x):
            if rng.random() > p_m:
                return x
            x_new = x.copy()
            for i in range(dim):
                if rng.random() < 1.0/dim:
                    delta1 = (x_new[i] - lo[i]) / (hi[i] - lo[i] + 1e-9)
                    delta2 = (hi[i] - x_new[i]) / (hi[i] - lo[i] + 1e-9)
                    u = rng.random()
                    if u < 0.5:
                        xy = (1.0 + 2.0 * (delta1 - 0.5)) ** (1.0 / (eta_m + 1.0))
                        deltaq = xy - 1.0
                    else:
                        xy = (1.0 + 2.0 * (0.5 - delta2)) ** (1.0 / (eta_m + 1.0))
                        deltaq = 1.0 - xy
                    x_new[i] = x_new[i] + deltaq * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
            
            front = [np.where(n_dom == 0)[0].tolist()]
            k = 0
            while front[k]:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                            ranks[j] = k + 1
                if next_front:
                    front.append(next_front)
                    k += 1
            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            dist = np.zeros(n)
            for rank in set(ranks):
                indices = np.where(ranks == rank)[0]
                if len(indices) <= 2:
                    dist[indices] = np.inf
                    continue
                for m in range(n_obj):
                    F_m = F[indices, m]
                    min_m = np.min(F_m)
                    max_m = np.max(F_m)
                    if max_m == min_m:
                        continue
                    sorted_indices = indices[np.argsort(F_m)]
                    dist[sorted_indices[0]] = np.inf
                    dist[sorted_indices[-1]] = np.inf
                    for i in range(1, len(sorted_indices)-1):
                        prev_idx = sorted_indices[i-1]
                        next_idx = sorted_indices[i+1]
                        dist[sorted_indices[i]] += (F[sorted_indices[i], m] - F[prev_idx, m]) / (max_m - min_m)
            return dist

        while used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            X_child = np.zeros((N, dim))
            F_child = np.zeros((N, n_obj))
            
            for i in range(N):
                # Select parents
                p1 = rng.integers(N)
                p2 = rng.integers(N)
                while p2 == p1:
                    p2 = rng.integers(N)
                
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                X_child[i] = c1
                F_child[i] = problem(c1)
                used += 1
                
                if used < budget and i + 1 < N:
                    X_child[i+1] = c2
                    F_child[i+1] = problem(c2)
                    used += 1
                    i += 1 # Skip next index in loop? No, better to handle carefully.
            
            # The above loop is tricky with i increment. Let's simplify:
            # Generate N offspring.
            # We need to be careful not to exceed budget.
            
            # Let's restart the generation logic to be safe.
            pass

        # Simplified Generation Loop
        # We will generate offspring in batches of 2 (from 1 crossover) or 1 (if budget is odd).
        
        # Re-initialize loop logic properly
        X_pop = X
        F_pop = F
        used = N
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Determine how many offspring we can make
            # We make pairs. If budget - used is odd, we make one single mutation.
            n_offspring = min(N, budget - used)
            
            X_off = np.zeros((n_offspring, dim))
            F_off = np.zeros((n_offspring, n_obj))
            
            for i in range(0, n_offspring, 2):
                p1 = rng.integers(N)
                p2 = rng.integers(N)
                while p2 == p1:
                    p2 = rng.integers(N)
                
                c1, c2 = sbx_crossover(X_pop[p1], X_pop[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                X_off[i] = c1
                F_off[i] = problem(c1)
                used += 1
                
                if i + 1 < n_offspring:
                    X_off[i+1] = c2
                    F_off[i+1] = problem(c2)
                    used += 1
            
            # Combine
            X_comb = np.vstack([X_pop, X_off])
            F_comb = np.vstack([F_pop, F_off])
            
            # Sort
            ranks = fast_non_dominated_sort(F_comb)
            dist = crowding_distance(F_comb, ranks)
            
            # Select
            # Sort by rank, then by distance (descending)
            sort_indices = np.lexsort((-dist, ranks))
            X_pop = X_comb[sort_indices[:N]]
            F_pop = F_comb[sort_indices[:N]]
            
        return X_pop
