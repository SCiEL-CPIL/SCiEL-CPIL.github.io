import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        max_evals = self.budget
        pop_size = 50
        
        # Initialize population
        X = np.random.uniform(lo, hi, (pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(F, n_total):
            ranks = np.zeros(n_total, dtype=int)
            S = [[] for _ in range(n_total)]
            n_dom = np.zeros(n_total, dtype=int)
            fronts = []
            
            for i in range(n_total):
                for j in range(n_total):
                    if i == j: continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])
                    
            k = 0
            while len(fronts) > k:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            
            return ranks, fronts

        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Main Loop
        while evals_used < max_evals:
            if time.time() > deadline:
                break
            
            # Differential Evolution Mutation
            # Generate offspring for each parent
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct parents
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i and r2 != i and r3 != i:
                        break
                
                # DE/rand/1
                F_factor = 0.5
                CR = 0.9
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        new_X[i, j] = v[j]
                    else:
                        new_X[i, j] = X[i, j]
                
                # Clip
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Evaluate
                new_F[i] = problem(new_X[i])
                evals_used += 1
                
            # Selection: Combine parent and offspring, keep best pop_size
            all_X = np.vstack([X, new_X])
            all_F = np.vstack([F, new_F])
            
            # Non-dominated sort
            n_total = len(all_F)
            ranks, fronts = non_dominated_sort(all_F, n_total)
            
            # Sort by rank, then by crowding distance (descending)
            # We need to pick pop_size best individuals.
            
            # Calculate crowding for each front
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to pick some from this front
                    remaining = pop_size - len(selected_indices)
                    # Calculate crowding for this front
                    dists = crowding_distance(all_F, front)
                    # Sort by dist descending
                    front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                    selected_indices.extend(front_sorted[:remaining])
                    break
                    
            # Update population
            X = all_X[selected_indices]
            F = all_F[selected_indices]
            
            # Local search on the best individual
            if evals_used < max_evals - 10:
                # Find the "best" individual (e.g., minimum sum of objectives)
                best_idx = np.argmin(np.sum(F, axis=1))
                best_x = X[best_idx].copy()
                best_f = F[best_idx].copy()
                
                # Simple coordinate local search with adaptive step size
                step_sizes = np.array([0.1, 0.05, 0.01]) * (hi - lo)
                
                for step_scale in [1.0, 0.5, 0.1]:
                    if evals_used >= max_evals:
                        break
                    for _ in range(5):
                        if evals_used >= max_evals or time.time() > deadline:
                            break
                        d = np.random.randint(self.dim)
                        step = step_sizes[d] * step_scale
                        # Try positive and negative step
                        for sign in [1, -1]:
                            if evals_used >= max_evals:
                                break
                            trial_x = best_x.copy()
                            trial_x[d] = np.clip(trial_x[d] + sign * step, lo[d], hi[d])
                            trial_f = problem(trial_x)
                            evals_used += 1
                            if dominates(trial_f, best_f):
                                best_x = trial_x
                                best_f = trial_f
                                break # Move on to next dimension
                    
                # Replace the best individual with the improved one
                X[best_idx] = best_x
                F[best_idx] = best_f

        return X
