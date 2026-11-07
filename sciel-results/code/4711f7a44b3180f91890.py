# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style evolutionary search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        range_ = hi - lo

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280

        # Phase 1: Initial population using Latin Hypercube Sampling
        pop_size = min(100, n // 4)
        if pop_size < 10:
            pop_size = min(10, n)
        
        # Generate LHS
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            X[:, d] = (np.arange(pop_size) + np.random.random(pop_size)) / pop_size
            X[:, d] = X[np.random.permutation(pop_size), d]
        X = lo + X * range_
        
        F = np.array([problem(x) for x in X])
        evals_used = pop_size

        # Helper functions
        def dominates(f1, f2):
            """Check if f1 dominates f2 (f1 <= f2 in all, f1 < f2 in at least one)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F):
            """
            Returns a list of lists, where each inner list contains indices of points
            in that non-dominated front. Front 0 is the best.
            """
            n_pts = len(F)
            S = [[] for _ in range(n_pts)]  # S[i] = set of points dominated by i
            n_dominated = np.zeros(n_pts, dtype=int)  # n_dominated[i] = count of points dominating i
            fronts = [[]]
            
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
            
            for i in range(n_pts):
                if n_dominated[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            # Remove the last empty front
            if len(fronts) > 0 and len(fronts[-1]) == 0:
                fronts.pop()
                
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for points in a front.
            Returns an array of distances for the points in front_indices.
            """
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            distances = np.zeros(n)
            F_front = F[front_indices]
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_indices = np.argsort(F_front[:, m])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                f_min = F_front[sorted_indices[0], m]
                f_max = F_front[sorted_indices[-1], m]
                
                if f_max - f_min < 1e-10:
                    continue
                    
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (F_front[sorted_indices[i + 1], m] - F_front[sorted_indices[i - 1], m]) / (f_max - f_min)
            
            return distances

        # Main Evolutionary Loop
        while evals_used < n:
            # Check time
            if time.time() > deadline:
                break

            # Create offspring
            offspring_size = min(pop_size, n - evals_used)
            
            # Tournament selection
            def select_tournament(F, fronts):
                # Assign rank
                rank = np.zeros(len(F), dtype=int)
                for i, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = i
                
                # Crowding distance per front
                cd = np.zeros(len(F))
                for front in fronts:
                    if len(front) > 0:
                        cd[front] = crowding_distance(F, front)
                
                # Tournament of 2
                i = np.random.randint(len(F))
                j = np.random.randint(len(F))
                
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                else:
                    return i if cd[i] > cd[j] else j

            parents = []
            for _ in range(offspring_size):
                p1 = select_tournament(F, fast_non_dominated_sort(F))
                p2 = select_tournament(F, fast_non_dominated_sort(F))
                parents.append((p1, p2))
            
            # Crossover and Mutation
            children = []
            for p1, p2 in parents:
                # Simulated Binary Crossover (SBX)
                x1 = X[p1]
                x2 = X[p2]
                
                # Uniform crossover for simplicity and speed, or SBX
                # Let's use a blend of uniform and mutation
                mask = np.random.random(dim) < 0.5
                child = np.where(mask, x1, x2)
                
                # Polynomial mutation
                # Simple Gaussian mutation with adaptive sigma
                sigma = 0.2 * range_
                child = child + sigma * np.random.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                children.append(child)
            
            children = np.array(children)
            
            # Evaluate children
            F_children = np.array([problem(x) for x in children])
            evals_used += len(children)
            
            # Combine parents and children
            X_all = np.vstack([X, children])
            F_all = np.vstack([F, F_children])
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_all)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    sorted_front = np.array(front)[np.argsort(cd)[::-1]]
                    for idx in sorted_front[:remaining]:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        return X
# EVOLVE-BLOCK-END
