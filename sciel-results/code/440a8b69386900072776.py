"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer with efficient vectorized operations."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size: large enough for diversity, small enough for budget
        N = 40  # Population size
        
        def _evaluate_batch(X_batch):
            """Evaluate a batch of points, respecting budget."""
            F_batch = np.array([problem(x) for x in X_batch])
            return F_batch
        
        # Initialize population with Latin hypercube for better initial spread
        X = np.zeros((N, dim))
        for i in range(N):
            for d in range(dim):
                # Use stratified sampling
                X[i, d] = lo[d] + (hi[d] - lo[d]) * ((i + np.random.random()) / N)
        
        F = _evaluate_batch(X)
        evals = N
        
        # NSGA-II Helper Functions
        def fast_non_dominated_sort(F):
            """Returns a list of fronts (indices) and rank array."""
            M = len(F)
            S = [set() for _ in range(M)]
            n = [0] * M
            rank = [0] * M
            
            for i in range(M):
                for j in range(i + 1, M):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n[i] += 1
                        
            fronts = []
            front_idx = 0
            current_front = [i for i in range(M) if n[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                for i in current_front:
                    rank[i] = front_idx
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            next_front.append(j)
                front_idx += 1
                current_front = next_front
                
            # Handle any unassigned points
            unassigned = [i for i in range(M) if rank[i] == 0 and i not in fronts[0]] if fronts else list(range(M))
            if unassigned:
                if fronts:
                    fronts[-1].extend(unassigned)
                    for i in unassigned:
                        rank[i] = len(fronts) - 1
                else:
                    fronts.append(unassigned)
                    for i in unassigned:
                        rank[i] = 0
                        
            return fronts, rank

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for points in a specific front."""
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
                
            dist = np.zeros(m)
            F_front = F[front_indices]
            
            for k in range(n_obj):
                sorted_idx = np.argsort(F_front[:, k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_range = F_front[sorted_idx[-1], k] - F_front[sorted_idx[0], k]
                if f_range == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (F_front[sorted_idx[i+1], k] - F_front[sorted_idx[i-1], k]) / f_range
                    
            return {front_indices[i]: dist[i] for i in range(m)}

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            eta_c = 15
            if np.random.random() < 0.95:  # High crossover rate
                child1 = np.empty(dim)
                child2 = np.empty(dim)
                for i in range(dim):
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = np.random.random()
                        if u <= 1.0 / (eta_c + 1.0):
                            beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                        elif u >= 1.0 - 1.0 / (eta_c + 1.0):
                            beta = (2.0 * (1.0 - u)) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta = (u / (1.0 + eta_c - eta_c * u)) ** (1.0 / (eta_c + 1.0))
                        
                        child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    else:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x):
            """Polynomial Mutation with bounded handling."""
            eta_m = 15
            child = x.copy()
            for i in range(dim):
                if np.random.random() < 1.0 / dim:  # Mutation rate per variable
                    lo_i = lo[i]
                    hi_i = hi[i]
                    range_i = hi_i - lo_i
                    if range_i == 0:
                        continue
                    
                    # Normalize to [0, 1]
                    y = (x[i] - lo_i) / range_i
                    y = np.clip(y, 0.0, 1.0)
                    
                    u = np.random.random()
                    if u < 0.5:
                        xy = 2.0 * u
                        delta = xy ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        xy = 2.0 * (1.0 - u)
                        delta = 1.0 - xy ** (1.0 / (eta_m + 1.0))
                    
                    child[i] = x[i] + delta * range_i
                    child[i] = np.clip(child[i], lo_i, hi_i)
            return child

        # Main NSGA-II Loop
        while evals < budget:
            if time.time() > deadline:
                break
                
            # Compute fronts and crowding for current population
            fronts, rank = fast_non_dominated_sort(F)
            dist = {}
            for front in fronts:
                d = crowding_distance(F, front)
                dist.update(d)
                
            # Selection function using NSGA-II selection
            def select_one():
                i = np.random.randint(N)
                j = np.random.randint(N)
                if rank[i] < rank[j]:
                    return i
                elif rank[i] > rank[j]:
                    return j
                else:
                    if dist[i] > dist[j]:
                        return i
                    else:
                        return j
                        
            # Generate offspring
            offspring = np.zeros((N, dim))
            for i in range(0, N, 2):
                p1_idx = select_one()
                p2_idx = select_one()
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring[i] = c1
                offspring[i+1] = c2
                
            # Evaluate offspring
            F_off = _evaluate_batch(offspring)
            evals += N
            
            # Combine parents and offspring
            X_combined = np.vstack((X, offspring))
            F_combined = np.vstack((F, F_off))
            
            # Sort combined population
            fronts_comb, rank_comb = fast_non_dominated_sort(F_combined)
            
            # Select next generation
            new_X = []
            new_F = []
            
            for front in fronts_comb:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Select from this front based on crowding distance
                    F_front = F_combined[front]
                    front_list = list(front)
                    d = crowding_distance(F_front, list(range(len(front))))
                    d_values = [d[i] for i in front_list]
                    sorted_indices = np.argsort(d_values)[::-1]
                    need = N - len(new_X)
                    for k in range(need):
                        idx = front_list[sorted_indices[k]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
# EVOLVE-BLOCK-END
