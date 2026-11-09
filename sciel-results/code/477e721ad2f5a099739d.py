import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        pop_size = 20
        if budget < pop_size:
            pop_size = budget
            
        # Initialization
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def non_dominated_sort(F):
            """Fast non-dominated sorting"""
            n = len(F)
            S = [set() for _ in range(n)]
            n_dom = [0] * n
            ranks = [0] * n
            front = [set()]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dominates(F[i], F[j]):
                        S[i].add(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    ranks[i] = 0
                    front[0].add(i)
                    
            k = 0
            while front[k]:
                next_front = set()
                for i in front[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = k + 1
                            next_front.add(j)
                k += 1
                if next_front:
                    front.append(next_front)
                else:
                    break
            return ranks, front
            
        def crowding_distance(F, indices):
            """Calculate crowding distance for a front"""
            n = len(indices)
            if n == 0: return []
            if n == 1: return [np.inf]
            
            dist = [0.0] * n
            for m in range(n_obj):
                vals = [F[i][m] for i in indices]
                min_val = min(vals)
                max_val = max(vals)
                if max_val == min_val:
                    for i in range(n):
                        dist[indices.index(i)] = np.inf # Approximation
                    continue
                # Sort by objective m
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n-1):
                    dist[sorted_idx[i]] += (F[sorted_idx[i+1]][m] - F[sorted_idx[i-1]][m]) / (max_val - min_val)
            return dist
            
        def select_parents(F, ranks, front):
            """Tournament selection based on rank and crowding"""
            # Simplified: Pick 2 random, compare rank, then crowding
            # For efficiency, we can just pick random non-dominated points
            # But let's do a proper tournament
            pass
            
        # Main Loop
        while used < budget:
            if time.time() - start_time > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = 10
            for _ in range(n_offspring):
                if used >= budget: break
                if time.time() - start_time > deadline: break
                
                # Select 2 parents
                # Simple: Random selection from population
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # SBX Crossover
                if np.random.rand() < 0.9: # Crossover prob
                    beta = 0.5 # Distribution index
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for i in range(dim):
                        if np.random.rand() < 0.5:
                            # SBX logic
                            u = np.random.rand()
                            if u <= 1/(beta+1):
                                beta_q = ((2*u)**(1/(beta+1)))
                            elif u >= 1-1/(beta+1):
                                beta_q = ((2*(1-u))**(1/(beta+1)))
                            else:
                                beta_q = 1
                            
                            x1, x2 = X[p1_idx][i], X[p2_idx][i]
                            if x1 == x2:
                                child1[i] = x1
                                child2[i] = x2
                            else:
                                child1[i] = 0.5 * ((1+beta_q)*x1 + (1-beta_q)*x2)
                                child2[i] = 0.5 * ((1+beta_q)*x2 + (1-beta_q)*x1)
                        else:
                            child1[i] = X[p1_idx][i]
                            child2[i] = X[p2_idx][i]
                            
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                    
                    # Evaluate
                    f1 = problem(child1)
                    f2 = problem(child2)
                    used += 2
                    offspring_F.append(f1)
                    offspring_F.append(f2)
                else:
                    # Mutation
                    child = X[p1_idx].copy()
                    # Gaussian mutation
                    sigma = 0.2
                    child += sigma * (hi - lo) * np.random.randn(dim)
                    child = np.clip(child, lo, hi)
                    offspring_X.append(child)
                    f = problem(child)
                    used += 1
                    offspring_F.append(f)
                    
            if not offspring_X:
                break
                
            # Combine
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.array([F] + [offspring_F]) # F is array, offspring_F is list of arrays
            
            # Non-dominated sort
            ranks, front = non_dominated_sort(F_new)
            
            # Select next population
            next_X = []
            next_F = []
            for i in front:
                if len(next_X) + len(i) <= pop_size:
                    # Add all
                    for idx in i:
                        next_X.append(X_new[idx])
                        next_F.append(F_new[idx])
                else:
                    # Need to select some from this front based on crowding
                    need = pop_size - len(next_X)
                    dists = crowding_distance(F_new, list(i))
                    # Sort indices by distance
                    sorted_indices = sorted(list(i), key=lambda idx: dists[list(i).index(idx)], reverse=True)
                    for idx in sorted_indices[:need]:
                        next_X.append(X_new[idx])
                        next_F.append(F_new[idx])
                    break
                    
            if len(next_X) < pop_size:
                # Fill with random if not enough (should not happen)
                while len(next_X) < pop_size:
                    x = lo + (hi - lo) * np.random.rand(dim)
                    f = problem(x)
                    used += 1
                    next_X.append(x)
                    next_F.append(f)
                    
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
