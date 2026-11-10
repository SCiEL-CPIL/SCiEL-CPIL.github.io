import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size: use 40, so 40 init + 4*40 = 200
        pop_size = 40
        # Adjust if budget doesn't divide evenly
        n_gen = (budget - pop_size) // pop_size
        if n_gen < 1:
            pop_size = max(4, budget // 2)
            n_gen = (budget - pop_size) // pop_size
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        pc = 0.9  # crossover probability
        nc = 30.0  # SBX distribution index
        pm = 1.0 / dim  # mutation probability per gene
        nm = 20.0  # polynomial mutation distribution index
        
        def dominates(a, b):
            """a dominates b if a <= b in all objectives and < in at least one"""
            return np.all(a <= b) and np.any(a < b)
        
        def fast_non_dominated_sort(F):
            """Return list of fronts, each front is a list of indices"""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            ndom = [0] * n  # number of solutions dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        ndom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        ndom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if ndom[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            return fronts, rank
        
        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front"""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            cd = [0.0] * n
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = F[front_indices[sorted_idx[0]]][m]
                f_max = F[front_indices[sorted_idx[-1]]][m]
                f_range = f_max - f_min
                
                if f_range < 1e-30:
                    continue
                    
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F[front_indices[sorted_idx[i+1]]][m] - 
                                          F[front_indices[sorted_idx[i-1]]][m]) / f_range
            
            return cd
        
        def sbx_crossover(x1, x2, lo, hi):
            """Simulated Binary Crossover"""
            if rng.random() > pc:
                return x1.copy(), x2.copy()
            
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(dim):
                if rng.random() < 0.5:
                    x1i, x2i = x1[i], x2[i]
                    if abs(x1i - x2i) < 1e-14:
                        continue
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (nc + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (nc + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1i + (1 - beta) * x2i)
                    child2[i] = 0.5 * ((1 - beta) * x1i + (1 + beta) * x2i)
                    
                    # Clip to bounds
                    child1[i] = np.clip(child1[i], lo[i], hi[i])
                    child2[i] = np.clip(child2[i], lo[i], hi[i])
            
            return child1, child2
        
        def polynomial_mutation(x, lo, hi):
            """Polynomial mutation"""
            for i in range(dim):
                if rng.random() < pm:
                    x_i = x[i]
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (nm + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (nm + 1))
                    
                    yl = (x_i - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.5
                    yr = (hi[i] - x_i) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.5
                    
                    x[i] += delta * max((yl + yr), 1e-10) ** (nm + 1) * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            
            return x
        
        # Main loop
        while used < budget and (time.time() - start_time) < deadline:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Evaluate current population's ranks for selection
            fronts, rank = fast_non_dominated_sort(F)
            
            # Compute crowding distance for each front
            cd = [0.0] * pop_size
            for front in fronts:
                front_cd = crowding_distance(F, front)
                for idx, cd_val in zip(front, front_cd):
                    cd[idx] = cd_val
            
            # Binary tournament selection
            def tournament():
                i, j = rng.integers(0, pop_size, size=2)
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    return i if cd[i] > cd[j] else j
            
            for _ in range(pop_size):
                p1 = tournament()
                p2 = tournament()
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                f1 = problem(c1)
                f2 = problem(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                used += 2
                
                if used >= budget or (time.time() - start_time) > deadline:
                    break
            
            if len(offspring_X) == 0:
                break
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts, rank = fast_non_dominated_sort(combined_F)
            
            # Compute crowding distance
            n_combined = len(combined_F)
            cd = [0.0] * n_combined
            for front in fronts:
                front_cd = crowding_distance(combined_F, front)
                for idx, cd_val in zip(front, front_cd):
                    cd[idx] = cd_val
            
            # Select next population
            new_idx = []
            for front in fronts:
                if len(new_idx) + len(front) <= pop_size:
                    new_idx.extend(front)
                else:
                    # Sort by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    new_idx.extend(front_sorted[:pop_size - len(new_idx)])
                    break
            
            X = combined_X[new_idx]
            F = combined_F[new_idx]
        
        return X
