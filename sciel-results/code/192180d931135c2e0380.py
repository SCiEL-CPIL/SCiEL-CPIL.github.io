import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # NSGA-II Parameters
        pop_size = 20
        n_gen = max(1, (budget - pop_size) // pop_size)
        
        # Initialize Population
        # Random uniform initialization
        X = np.random.rand(pop_size, dim) * (hi - lo) + lo
        F = np.array([problem(x) for x in X])
        
        used = pop_size
        
        # NSGA-II Loop
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            if used + pop_size > budget:
                break
                
            # Generate Offspring
            offspring = np.zeros((pop_size, dim))
            
            for i in range(0, pop_size, 2):
                # Crossover
                if np.random.rand() < 0.9: # Crossover probability
                    # Simulated Binary Crossover (SBX)
                    eta_c = 20.0
                    u = np.random.rand(dim)
                    beta = np.zeros(dim)
                    beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1.0 / (eta_c + 1))
                    beta[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (1.0 / (eta_c + 1))
                    
                    x1 = X[i]
                    x2 = X[(i + 1) % pop_size]
                    
                    c1 = 0.5 * (x1 + x2 + beta * (x1 - x2))
                    c2 = 0.5 * (x1 + x2 - beta * (x1 - x2))
                    
                    # Clip to bounds
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)
                    
                    # Mutation
                    # Polynomial Mutation
                    eta_m = 20.0
                    for k in range(dim):
                        if np.random.rand() < (1.0 / dim):
                            u = np.random.rand()
                            delta = 0.0
                            if u < 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            range_val = hi[k] - lo[k]
                            c1[k] += delta * range_val
                            c1[k] = np.clip(c1[k], lo[k], hi[k])
                        
                        if np.random.rand() < (1.0 / dim):
                            u = np.random.rand()
                            delta = 0.0
                            if u < 0.5:
                                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            range_val = hi[k] - lo[k]
                            c2[k] += delta * range_val
                            c2[k] = np.clip(c2[k], lo[k], hi[k])
                    
                    offspring[i] = c1
                    offspring[(i + 1) % pop_size] = c2
                else:
                    offspring[i] = X[i].copy()
                    offspring[(i + 1) % pop_size] = X[(i + 1) % pop_size].copy()
            
            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine Parents and Offspring
            X_combined = np.vstack((X, offspring))
            F_combined = np.vstack((F, F_offspring))
            
            # Selection: Fast Non-Dominated Sorting and Crowding Distance
            # 1. Non-Dominated Sorting
            fronts = self._fast_non_dominated_sort(F_combined)
            
            # 2. Selection
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            current_front_idx = 0
            selected_count = 0
            
            while selected_count < pop_size and current_front_idx < len(fronts):
                front_indices = fronts[current_front_idx]
                
                if selected_count + len(front_indices) <= pop_size:
                    # Add entire front
                    for idx in front_indices:
                        new_X[selected_count] = X_combined[idx]
                        new_F[selected_count] = F_combined[idx]
                        selected_count += 1
                    current_front_idx += 1
                else:
                    # Need to select from this front based on crowding distance
                    front_F = F_combined[front_indices]
                    crowding = self._crowding_distance(front_F)
                    
                    # Sort by crowding distance in descending order
                    sorted_indices = np.argsort(-crowding)
                    
                    needed = pop_size - selected_count
                    for j in range(needed):
                        idx = front_indices[sorted_indices[j]]
                        new_X[selected_count] = X_combined[idx]
                        new_F[selected_count] = F_combined[idx]
                        selected_count += 1
                    break
            
            X = new_X
            F = new_F

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [set() for _ in range(n)]  # Dominated sets
        n_dom = [0] * n  # Number of solutions dominating i
        fronts = [set() for _ in range(n)]  # Fronts
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                # i dominates j if F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        # First front
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].add(i)
        
        k = 1
        while len(fronts[k - 1]) > 0 and k < n:
            next_front = set()
            for i in fronts[k - 1]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            fronts[k] = next_front
            k += 1
        
        # Convert to lists for easier indexing
        return [list(front) for front in fronts if len(front) > 0]

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort indices by objective k
            idx_sorted = np.argsort(F[:, k])
            dist[idx_sorted[0]] = np.inf
            dist[idx_sorted[-1]] = np.inf
            
            f_range = F[idx_sorted[-1], k] - F[idx_sorted[0], k]
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                dist[idx_sorted[i]] += (F[idx_sorted[i + 1], k] - F[idx_sorted[i - 1], k]) / f_range
        
        return dist
