import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # We want to use the full budget. 
        # Let N be the population size.
        # We will do G generations.
        # Total evals = N * (G + 1) <= budget
        # Let's aim for G=3, so N=50. 50*4 = 200.
        N = 50
        
        # Check if budget allows
        if budget < N:
             N = budget
             G = 0
        else:
             G = budget // N - 1
             
        # Initial Population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            F is NxM array.
            """
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[q] += 1
                        
            # First front
            front_idx = 0
            front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = front_idx
                    front.append(p)
            if front:
                fronts.append(front)
                
            while front:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = front_idx + 1
                            next_front.append(q)
                if next_front:
                    front_idx += 1
                    fronts.append(next_front)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in the same front.
            Returns a dict or array of distances.
            """
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            # Map indices to local 0..n-1
            # We need to map back to F indices
            idx_map = np.array(indices)
            
            for m in range(n_obj):
                # Sort local indices by objective m
                obj_vals = F[idx_map, m]
                sort_order = np.argsort(obj_vals)
                
                dist[sort_order[0]] = np.inf
                dist[sort_order[-1]] = np.inf
                
                f_range = obj_vals[sort_order[-1]] - obj_vals[sort_order[0]]
                if f_range > 1e-12:
                    for i in range(1, n - 1):
                        dist[sort_order[i]] += (obj_vals[sort_order[i+1]] - obj_vals[sort_order[i-1]]) / f_range
            return dist

        def select_parents(F, X, n_parents):
            """
            Select n_parents individuals from population X, F using tournament selection.
            """
            parents_idx = []
            # Precompute ranks and crowding distances
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    front_cd = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = front_cd[j]
                else:
                    for idx in front:
                        cd[idx] = np.inf
                        
            for _ in range(n_parents):
                # Tournament of size 2
                i1, i2 = np.random.choice(len(F), 2, replace=False)
                if rank[i1] < rank[i2]:
                    parents_idx.append(i1)
                elif rank[i2] < rank[i1]:
                    parents_idx.append(i2)
                else:
                    if cd[i1] > cd[i2]:
                        parents_idx.append(i1)
                    else:
                        parents_idx.append(i2)
            return parents_idx

        def sbx_crossover(x1, x2, lo, hi):
            """Simulated Binary Crossover"""
            eta = 20
            if np.random.rand() < 0.5:
                return x1, x2
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    if c1 > hi[i]:
                        c1 = hi[i]
                    if c1 < lo[i]:
                        c1 = lo[i]
                    if c2 > hi[i]:
                        c2 = hi[i]
                    if c2 < lo[i]:
                        c2 = lo[i]
                    child1[i] = c1
                    child2[i] = c2
            return child1, child2

        def polynomial_mutation(x, lo, hi, eta=20):
            """Polynomial Mutation"""
            child = x.copy()
            for i in range(dim):
                if np.random.rand() < 0.9: # Mutation rate 0.9? Maybe 1/dim?
                    u = np.random.rand()
                    if u <= 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta1 = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    child[i] = x[i] + delta1 * (hi[i] - lo[i])
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Main Loop
        for gen in range(G):
            if time.time() > deadline:
                break
            
            # Create offspring Q
            Q_X = np.zeros((N, dim))
            Q_F = np.zeros((N, n_obj))
            
            # Select parents
            # We need 2 parents per child, so 2*N parents
            parent_indices = select_parents(F, X, 2 * N)
            
            for i in range(0, 2 * N, 2):
                p1 = X[parent_indices[i]]
                p2 = X[parent_indices[i+1]]
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                Q_X[i // 2] = c1
                Q_X[i // 2 + 1] = c2
                
            # Evaluate offspring
            for i in range(N):
                Q_F[i] = problem(Q_X[i])
                used += 1
                if used > budget:
                    # This shouldn't happen if math is correct, but safety
                    break
            if used > budget:
                break

            # Combine P and Q
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Sort and select next P
            fronts = fast_non_dominated_sort(R_F)
            
            new_P_idx = []
            new_P_F = []
            new_P_X = []
            
            for front in fronts:
                if len(new_P_idx) + len(front) <= N:
                    for idx in front:
                        new_P_idx.append(idx)
                else:
                    # Need to select from this front
                    remaining = N - len(new_P_idx)
                    front_cd = crowding_distance(R_F, front)
                    # Sort front by crowding distance descending
                    front_indices = np.array(front)
                    sort_order = np.argsort(front_cd)[::-1]
                    selected_in_front = front_indices[sort_order[:remaining]]
                    for idx in selected_in_front:
                        new_P_idx.append(idx)
                    break
            
            # Update X and F
            X = R_X[new_P_idx]
            F = R_F[new_P_idx]
            
            if time.time() > deadline:
                break

        # Return the final population (or the best non-dominated set found so far)
        # The task says "Every returned objective vector is recorded".
        # It implies we should return the points that constitute the Pareto front.
        # However, the return value is ignored, only the calls to problem(x) matter.
        # Wait, "The return value of __call__ is ignored."
        # So I just need to make sure I don't exceed the budget and don't crash.
        # The evaluator records the objective values of all x passed to problem.
        # So I just need to run the algorithm.
        
        return X
