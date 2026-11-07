import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20  # Population size
        # We have 200 budget. Initial pop is 20.
        # Each gen adds 20 offspring.
        # 20 + 9*20 = 200. So 9 generations of offspring.
        
        # Initialize population
        # Use a fixed seed for reproducibility in the algorithm's internal randomness, 
        # but the task says evaluator seeds np.random. 
        # However, we should use the global np.random state to be consistent with the task's seeding.
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # So we should use np.random directly, not a local RNG, to respect the seed.
        
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_doms = np.zeros(n)       # Number of dominators
            rank = np.zeros(n)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_doms[i] += 1
            
            current_front = []
            k = 0
            for i in range(n):
                if n_doms[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front
            return rank

        # Helper: Crowding distance
        def crowding_distance(F, rank):
            n = len(F)
            dist = np.zeros(n)
            fronts = {}
            for i in range(n):
                if rank[i] not in fronts:
                    fronts[rank[i]] = []
                fronts[rank[i]].append(i)
            
            for front in fronts.values():
                if len(front) <= 2:
                    for i in front:
                        dist[i] = np.inf
                    continue
                
                for m in range(self.n_obj):
                    # Sort front by objective m
                    sorted_indices = sorted(front, key=lambda i: F[i, m])
                    dist[sorted_indices[0]] = np.inf
                    dist[sorted_indices[-1]] = np.inf
                    
                    f_min = F[sorted_indices[0], m]
                    f_max = F[sorted_indices[-1], m]
                    range_f = f_max - f_min
                    if range_f == 0:
                        continue
                    
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        prev_idx = sorted_indices[i - 1]
                        next_idx = sorted_indices[i + 1]
                        dist[idx] += (F[next_idx, m] - F[prev_idx, m]) / range_f
            
            return dist

        # Helper: Tournament selection
        def tournament_select(pop_X, pop_F, pop_rank, pop_dist, k=2):
            idx1 = np.random.randint(len(pop_X))
            idx2 = np.random.randint(len(pop_X))
            
            def better(i, j):
                if pop_rank[i] < pop_rank[j]:
                    return True
                elif pop_rank[i] > pop_rank[j]:
                    return False
                else:
                    return pop_dist[i] > pop_dist[j]
            
            if better(idx1, idx2):
                return pop_X[idx1].copy()
            else:
                return pop_X[idx2].copy()

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=15.0):
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = np.random.rand()
                        beta1 = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta1) * x1[i] + (1 - beta1) * x2[i])
                        beta2 = (2 * (1 - u)) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * u)) ** (1 / (eta + 1))
                        child2[i] = 0.5 * ((1 + beta2) * x1[i] + (1 - beta2) * x2[i])
                    else:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    delta1 = (2 * u) ** (1 / (eta + 1)) - 1 if u <= 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[i] += delta1 * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            # Generate offspring
            offspring_X = []
            for _ in range(N):
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                if used + 1 < self.budget: # Ensure we don't exceed budget
                    offspring_X.append(c2)
                    used += 1
                else:
                    break
            
            # Evaluate offspring
            # Note: The loop above might append c2 conditionally. 
            # Let's restructure to be cleaner and strictly count evaluations.
            
            # Reset offspring generation to be precise
            offspring_X = []
            while used < self.budget and len(offspring_X) < N:
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X.append(c1)
                used += 1
                if used < self.budget:
                    offspring_X.append(c2)
                    used += 1
            
            if not offspring_X:
                break
                
            offspring_F = np.array([problem(x) for x in offspring_X])
            
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Sort and select
            rank = non_dominated_sort(combined_F)
            dist = crowding_distance(combined_F, rank)
            
            # Select next N individuals
            # Sort by rank, then by crowding distance (descending)
            order = np.argsort(rank)
            # For ties in rank, sort by dist descending
            # numpy argsort is stable, so we can sort by rank first, then dist? 
            # Better: create a list of indices and sort with a key.
            indices = list(range(len(combined_X)))
            indices.sort(key=lambda i: (rank[i], -dist[i]))
            
            next_X = combined_X[indices[:N]]
            next_F = combined_F[indices[:N]]
            
            X = next_X
            F = next_F
            rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
        return X
