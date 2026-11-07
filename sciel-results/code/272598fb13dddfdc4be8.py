"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Initial population size
        # Budget is 200. We need to balance exploration and exploitation.
        # A population size of 20-40 is reasonable. Let's use 20 to allow ~10 generations.
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Random initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """Returns list of fronts, each containing indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = [0] * n            # number of individuals dominating i
            rank = [0] * n              # rank of each individual
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # q dominates p?
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
            
            # First front
            front = []
            for i in range(n):
                if n_doms[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            fronts = [front]
            k = 0
            while fronts[k]:
                next_front = []
                for p in fronts[k]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            rank[q] = k + 1
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                    k += 1
            return fronts, rank
        
        def crowding_distance(F, front_indices):
            """Calculates crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            # Map front indices to 0..n-1 for easier calculation
            F_front = F[front_indices]
            dist = [0.0] * n
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_indices = np.argsort(F_front[:, m])
                # Set boundary distances to infinity
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F_front[sorted_indices[0], m]
                f_max = F_front[sorted_indices[-1], m]
                range_m = f_max - f_min
                
                if range_m == 0:
                    continue
                    
                for i in range(1, n - 1):
                    # Accumulate distance
                    dist[sorted_indices[i]] += (F_front[sorted_indices[i+1], m] - F_front[sorted_indices[i-1], m]) / range_m
            
            return dist

        def select_parents(F, X, rank, n_parents):
            """Binary tournament selection based on rank and crowding distance."""
            parents = []
            # Pre-calculate crowding distances for all fronts
            fronts, _ = fast_non_dominated_sort(F)
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]
            
            for _ in range(n_parents):
                p1 = np.random.randint(len(F))
                p2 = np.random.randint(len(F))
                
                # Compare p1 and p2
                if rank[p1] < rank[p2]:
                    winner = p1
                elif rank[p2] < rank[p1]:
                    winner = p2
                else:
                    if crowding[p1] > crowding[p2]:
                        winner = p1
                    else:
                        winner = p2
                parents.append(X[winner])
            return parents

        def sbx_crossover(x1, x2, eta=15.0):
            """Simulated Binary Crossover."""
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    x1_i, x2_i = x1[i], x2[i]
                    if x1_i == x2_i:
                        continue
                    if x1_i > x2_i:
                        x1_i, x2_i = x2_i, x1_i
                    
                    if np.random.random() <= 0.5:
                        u = np.random.random()
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
                        c2[i] = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
                    else:
                        u = np.random.random()
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1 + beta) * x2_i + (1 - beta) * x1_i)
                        c2[i] = 0.5 * ((1 - beta) * x2_i + (1 + beta) * x1_i)
            return c1, c2

        def polynomial_mutation(x, eta=20.0):
            """Polynomial Mutation."""
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine number of children to generate
            # We can generate up to pop_size children, but limited by remaining budget
            n_children = min(pop_size, self.budget - evals)
            
            # Select parents
            # We need rank for selection. Recalculate if population changed significantly? 
            # For simplicity in this loop, we recalculate rank every generation.
            fronts, rank = fast_non_dominated_sort(F)
            parents = select_parents(F, X, rank, n_children)
            
            # Generate children
            X_new = []
            F_new = []
            
            for i in range(n_children):
                # Crossover
                p1 = parents[i]
                # Select second parent
                p2_idx = np.random.randint(len(parents))
                p2 = parents[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                if evals + 1 <= self.budget and time.time() < deadline:
                    f1 = problem(c1)
                    evals += 1
                    X_new.append(c1)
                    F_new.append(f1)
                else:
                    break
                    
                if evals + 1 <= self.budget and time.time() < deadline:
                    f2 = problem(c2)
                    evals += 1
                    X_new.append(c2)
                    F_new.append(f2)
                else:
                    break
            
            if not X_new:
                break
                
            X_new = np.array(X_new)
            F_new = np.array(F_new)
            
            # Combine parents and children
            X_combined = np.vstack([X, X_new])
            F_combined = np.vstack([F, F_new])
            
            # Selection: Keep best pop_size individuals
            fronts, rank = fast_non_dominated_sort(F_combined)
            
            # Sort fronts
            sorted_fronts = sorted(fronts, key=lambda x: len(x) if False else min([rank[i] for i in x]))
            # Actually, fronts are already sorted by rank in fast_non_dominated_sort? 
            # My implementation returns fronts in order of rank. Let's verify.
            # Yes, fronts[0] is rank 0, fronts[1] is rank 1, etc.
            
            X_next = []
            F_next = []
            
            for front in fronts:
                if len(X_next) + len(front) <= pop_size:
                    # Accept whole front
                    for idx in front:
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                else:
                    # Need to select subset of this front based on crowding distance
                    remaining = pop_size - len(X_next)
                    cd = crowding_distance(F_combined, front)
                    # Sort by crowding distance descending
                    sorted_front_indices = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front_indices[:remaining]:
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                    break
                    
            if not X_next:
                # Fallback if something went wrong
                X_next = list(X)
                F_next = list(F)
            else:
                X = np.array(X_next)
                F = np.array(F_next)

        return X
# EVOLVE-BLOCK-END
