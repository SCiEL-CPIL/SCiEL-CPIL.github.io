import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 320  # Safety margin

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = 20
        # Number of generations
        # Total budget = pop_size (init) + n_gen * pop_size (offspring)
        # 200 = 20 + 9 * 20 = 200
        n_gen = (self.budget - pop_size) // pop_size
        
        # Initialize population
        X = np.random.uniform(lo, hi, (pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = np.zeros(n, dtype=int)  # Domination count
            rank = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                        
            current_front = []
            for p in range(n):
                if n_d[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
                    
            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                current_front = next_front
            return rank

        # Helper: Crowding distance
        def crowding_distance(F, rank, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                # Sort front by m-th objective
                sorted_indices = np.argsort(F[front_indices, m])
                front_sorted = [front_indices[i] for i in sorted_indices]
                cd_local = np.zeros(n)
                cd_local[0] = np.inf
                cd_local[-1] = np.inf
                for i in range(1, n - 1):
                    cd_local[i] = (F[front_sorted[i+1], m] - F[front_sorted[i-1], m]) / (F[front_sorted[-1], m] - F[front_sorted[0], m] + 1e-10)
                # Map back to local indices
                for i, idx in enumerate(front_sorted):
                    local_idx = front_indices.index(idx)
                    cd[local_idx] += cd_local[i]
            return cd

        # Helper: Selection
        def select_parents(pop_X, pop_F, rank, cd, k=2):
            parents = []
            for _ in range(k):
                i, j = np.random.randint(0, len(pop_X), 2)
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[i] > rank[j]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        # Helper: Crossover (SBX)
        def sbx_crossover(x1, x2, eta=20):
            child1 = np.copy(x1)
            child2 = np.copy(x2)
            for i in range(dim):
                if x1[i] == x2[i]:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Helper: Mutation
        def polynomial_mutation(x, eta=20, prob=0.1):
            x = np.copy(x)
            for i in range(dim):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    delta1 = (2 * u) ** (1 / (eta + 1)) - 1
                    delta2 = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    if u < 0.5:
                        delta = delta1
                    else:
                        delta = delta2
                    x[i] = x[i] + (hi[i] - lo[i]) * delta
            x = np.clip(x, lo, hi)
            return x

        # Main Evolution Loop
        for gen in range(n_gen):
            if time.time() > self.deadline:
                break
                
            # Calculate rank and crowding distance for current population
            rank = non_dominated_sort(F)
            # Group by rank
            fronts = {}
            for i in range(pop_size):
                if rank[i] not in fronts:
                    fronts[rank[i]] = []
                fronts[rank[i]].append(i)
            
            cd = np.zeros(pop_size)
            for r in sorted(fronts.keys()):
                front_indices = fronts[r]
                cd[front_indices] = crowding_distance(F, rank, front_indices)
            
            # Generate offspring
            offspring_X = []
            for _ in range(pop_size // 2):
                p1, p2 = select_parents(X, F, rank, cd, 2)
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine
            X_new = np.vstack((X, offspring_X))
            F_new = np.vstack((F, offspring_F))
            
            # Select next generation
            rank_new = non_dominated_sort(F_new)
            fronts_new = {}
            for i in range(len(X_new)):
                if rank_new[i] not in fronts_new:
                    fronts_new[rank_new[i]] = []
                fronts_new[rank_new[i]].append(i)
            
            next_X = []
            next_F = []
            for r in sorted(fronts_new.keys()):
                front_indices = fronts_new[r]
                if len(next_X) + len(front_indices) <= pop_size:
                    next_X.extend([X_new[i] for i in front_indices])
                    next_F.extend([F_new[i] for i in front_indices])
                else:
                    # Need to select some from this front based on crowding distance
                    cd_front = crowding_distance(F_new, rank_new, front_indices)
                    # Sort by cd descending
                    sorted_front = sorted(front_indices, key=lambda i: cd_front[front_indices.index(i)], reverse=True)
                    needed = pop_size - len(next_X)
                    for i in range(needed):
                        idx = sorted_front[i]
                        next_X.append(X_new[idx])
                        next_F.append(F_new[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
