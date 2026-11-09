import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        rng = np.random.default_rng(0)
        
        # Population size
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)
            
        # Initialize population
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    # i dominates j if F[i] <= F[j] and F[i] != F[j]
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            fronts.append(current_front)
            k = 0
            while len(current_front) > 0:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                if len(current_front) > 0:
                    fronts.append(current_front)
                k += 1
            return fronts, rank

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_indices = np.argsort(F_sub[:, k])
                dist[sorted_indices[0]] = np.inf
                dist[sorted_indices[-1]] = np.inf
                
                f_min = F_sub[sorted_indices[0], k]
                f_max = F_sub[sorted_indices[-1], k]
                
                if f_max == f_min:
                    continue
                    
                for i in range(1, m - 1):
                    idx_curr = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    
                    # Calculate distance in objective k space
                    d = (F_sub[idx_next, k] - F_sub[idx_prev, k]) / (f_max - f_min)
                    dist[idx_curr] += d
                    
            return dist

        # Helper: SBX Crossover
        def sbx(x1, x2, lo, hi, eta_c=15.0):
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = x2[i], x1[i]
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1.0))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    # Clip
                    child1[i] = np.clip(child1[i], lo[i], hi[i])
                    child2[i] = np.clip(child2[i], lo[i], hi[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def poly_mutate(x, lo, hi, eta_m=20.0):
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1.0))
                    
                    x[i] = x[i] + delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Evolution Loop
        while used < self.budget:
            if time.time() >= deadline:
                break
                
            # Generate offspring
            offspring_X = np.empty((N, self.dim))
            for i in range(N // 2):
                # Select parents using tournament
                # Simple tournament of size 2
                p1_idx = rng.integers(0, N)
                p2_idx = rng.integers(0, N)
                
                # Rank-based selection? Or just random? 
                # Standard NSGA-II uses binary tournament on fitness (rank, crowding)
                # Let's implement a simple selection helper
                def select_parent():
                    a, b = rng.integers(0, N, 2)
                    # Get ranks and crowding distances for current pop
                    # We need to recompute ranks/dist for selection?
                    # To save time, we can precompute them outside the loop if we do it every generation.
                    # But we are inside the offspring generation loop.
                    # It's better to compute ranks/dist once per generation before generating offspring.
                    pass
                # Let's restructure: Compute fronts/dist for current X, F
                fronts, rank = non_dominated_sort(F)
                # Compute crowding distance for each front
                cd = np.zeros(N)
                for front in fronts:
                    if len(front) > 0:
                        cd[front] = crowding_distance(F, front)
                
                def compare_fitness(i, j):
                    if rank[i] != rank[j]:
                        return rank[i] < rank[j]
                    return cd[i] > cd[j]
                
                p1_idx = rng.integers(0, N)
                p2_idx = rng.integers(0, N)
                if not compare_fitness(p1_idx, p2_idx):
                    p1_idx, p2_idx = p2_idx, p1_idx
                    # Swap to ensure p1 is better? No, tournament picks the better one.
                    # Let's just pick the better of the two randomly selected.
                    p1_idx, p2_idx = rng.integers(0, N, 2)
                    if compare_fitness(p1_idx, p2_idx):
                        best = p1_idx
                    else:
                        best = p2_idx
                    # Actually, we need two parents. Let's do two tournaments.
                    # Parent 1
                    p1_a, p1_b = rng.integers(0, N, 2)
                    parent1 = p1_a if compare_fitness(p1_a, p1_b) else p1_b
                    # Parent 2
                    p2_a, p2_b = rng.integers(0, N, 2)
                    parent2 = p2_a if compare_fitness(p2_a, p2_b) else p2_b
                    
                    c1, c2 = sbx(X[parent1], X[parent2], lo, hi)
                    c1 = poly_mutate(c1, lo, hi)
                    c2 = poly_mutate(c2, lo, hi)
                    offspring_X[2*i] = c1
                    offspring_X[2*i+1] = c2
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Sort combined population
            fronts, rank = non_dominated_sort(F_all)
            cd_all = np.zeros(len(X_all))
            for front in fronts:
                if len(front) > 0:
                    cd_all[front] = crowding_distance(F_all, front)
            
            # Select next N parents
            new_X = np.empty((N, self.dim))
            new_F = np.empty((N, self.n_obj))
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Sort front by crowding distance (descending)
                    front_sorted = np.sort(front, key=lambda i: cd_all[i], reverse=True)
                    needed = N - len(selected_indices)
                    selected_indices.extend(front_sorted[:needed])
                    break
            
            X = X_all[selected_indices]
            F = F_all[selected_indices]
            
        return X
