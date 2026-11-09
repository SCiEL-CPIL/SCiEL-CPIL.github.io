import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        # With budget 200, N=20 gives 10 generations.
        N = 20
        
        # Initial population
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Parameters for SBX and Mutation
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Probability of crossover
        p_m = 1.0 / dim # Probability of mutation per variable
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            front = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1
                
                if n_dominates[i] == 0:
                    front.append(i)
            
            fronts = []
            current_front = front
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n == 0:
                return []
            if n == 1:
                return [float('inf')]
            
            dist = [0.0] * n
            for k in range(n_obj):
                obj_values = np.array([F[i][k] for i in indices])
                min_val = np.min(obj_values)
                max_val = np.max(obj_values)
                if max_val - min_val < 1e-10:
                    continue
                # Sort indices by objective k
                sorted_idx = np.argsort(obj_values)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def sbx_crossover(x1, x2, lo, hi):
            if np.random.rand() > p_c:
                return x1.copy(), x2.copy()
            
            x1_new = x1.copy()
            x2_new = x2.copy()
            
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] < x2[i]:
                        x_min, x_max = x1[i], x2[i]
                    else:
                        x_min, x_max = x2[i], x1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    c1 = np.clip(c1, lo[i], hi[i])
                    c2 = np.clip(c2, lo[i], hi[i])
                    
                    x1_new[i] = c1
                    x2_new[i] = c2
            
            return x1_new, x2_new

        def polynomial_mutation(x, lo, hi):
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < p_m:
                    if x[i] == lo[i] or x[i] == hi[i]:
                        continue
                    u = np.random.rand()
                    if u <= 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta2 = (1 - 2 * (1 - u)) ** (1.0 / (eta_m + 1)) - 1
                        delta1 = -delta2
                    
                    x_new[i] = x[i] + delta1 * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = []
            for _ in range(N // 2):
                # Tournament selection (size 2)
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                # Simple selection: pick one, then pick another, compare dominance
                # For efficiency in small pop, just random pairs
                p1 = X[idx1]
                p2 = X[idx2]
                
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, F_offspring])
            
            # Sort
            fronts = fast_non_dominated_sort(F_combined)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = N - len(new_X)
                    dists = crowding_distance(F_combined, front)
                    # Sort by dist descending
                    sorted_indices = np.argsort(dists)[::-1]
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
