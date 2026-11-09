import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random
        
        # Population size
        # Use a size that allows several generations
        # 20 is a good balance for 200 budget
        pop_size = 20
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            front = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
            
            for p in range(n):
                if n_dominates[p] == 0:
                    front[0].append(p)
            
            i = 0
            while front[i]:
                next_front = []
                for p in front[i]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            next_front.append(q)
                i += 1
                front.append(next_front)
            
            # Remove empty last front if any
            if not front[-1]:
                front.pop()
                
            return front

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            m = F.shape[1]
            
            for k in range(m):
                obj_values = F[front_indices, k]
                sorted_indices = np.argsort(obj_values)
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                if obj_values[sorted_indices[-1]] == obj_values[sorted_indices[0]]:
                    continue
                
                for i in range(1, n - 1):
                    dist[sorted_indices[i]] += (obj_values[sorted_indices[i+1]] - obj_values[sorted_indices[i-1]]) / (obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]])
            
            return dist

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, prob=0.9, eta=20.0):
            if rng.random() < prob:
                child1 = np.zeros_like(x1)
                child2 = np.zeros_like(x2)
                for i in range(len(x1)):
                    if x1[i] == x2[i]:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, prob=1.0/dim, eta=20.0):
            child = x.copy()
            for i in range(len(x)):
                if rng.random() < prob:
                    u = rng.random()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    child[i] += delta1 * (hi[i] - lo[i])
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Main Loop
        while used < budget:
            # Create offspring
            offspring = []
            for i in range(pop_size):
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                while p2_idx == p1_idx:
                    p2_idx = rng.integers(pop_size)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring.append(c1)
                offspring.append(c2)
                
                if len(offspring) >= pop_size:
                    break
            
            # If we didn't generate enough, just duplicate some (should not happen with pop_size=20 and pairs)
            # Actually, the loop above generates 2 per iteration, so for pop_size=20, we need 10 iterations.
            # Let's just generate exactly pop_size offspring.
            
            # Re-do offspring generation cleanly
            offspring = []
            for i in range(0, pop_size, 2):
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                while p2_idx == p1_idx:
                    p2_idx = rng.integers(pop_size)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring.append(c1)
                if i + 1 < pop_size:
                    offspring.append(c2)

            X_off = np.array(offspring)
            F_off = np.array([problem(x) for x in X_off])
            used += pop_size
            
            # Combine
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Sort and Select
            fronts = fast_non_dominated_sort(F_comb)
            
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding
                    front_F = F_comb[front]
                    dists = crowding_distance(front_F, np.arange(len(front)))
                    sorted_front = np.argsort(dists)[::-1]
                    needed = pop_size - len(new_X)
                    for j in range(needed):
                        idx_in_front = sorted_front[j]
                        global_idx = front[idx_in_front]
                        new_X.append(X_comb[global_idx])
                        new_F.append(F_comb[global_idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            if used >= budget:
                break

        return X
