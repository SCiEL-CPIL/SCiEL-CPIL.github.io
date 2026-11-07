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
        range_ = hi - lo
        
        def time_check():
            return time.time() < deadline
        
        def non_dominated_sort(F):
            """NSGA-II non-dominated sorting"""
            n = len(F)
            fronts = [[]]
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            for i in range(n):
                if domination_count[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            return fronts
        
        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices"""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            n_obj = self.n_obj
            F_sub = F[indices]
            
            for m in range(n_obj):
                obj_vals = F_sub[:, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-10:
                    continue
                order = np.argsort(obj_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n-1):
                    dist[order[i]] += (obj_vals[order[i+1]] - obj_vals[order[i-1]]) / (max_val - min_val)
            
            return dist
        
        def differential_mutation(X, F, pop_size, lo, hi, range_, evals, problem, deadline):
            """Adaptive differential mutation for WFG9"""
            if len(X) < 3:
                # Fallback to random mutation if population too small
                x_base = np.random.uniform(lo, hi)
                for i in range(self.dim):
                    if np.random.rand() < 1.0 / self.dim:
                        x_base[i] += np.random.normal(0, 0.1 * range_[i])
                x_base = np.clip(x_base, lo, hi)
                return x_base
            
            # Select base and two other individuals
            base_idx = np.random.randint(pop_size)
            other_indices = np.random.choice(pop_size, 2, replace=False)
            while base_idx in other_indices:
                other_indices = np.random.choice(pop_size, 2, replace=False)
            
            x_base = X[base_idx]
            x1 = X[other_indices[0]]
            x2 = X[other_indices[1]]
            
            # Adaptive scaling factor based on convergence
            # Estimate convergence from objective spread
            F_spread = np.std(F, axis=0)
            convergence = 1.0 / (1.0 + np.sum(F_spread))
            F_scale = 0.5 + 0.5 * convergence  # Between 0.5 and 1.0
            
            # Differential mutation: x_trial = x_base + F_scale * (x1 - x2)
            x_trial = x_base + F_scale * (x1 - x2)
            
            # Add small random perturbation to escape local optima
            perturbation = 0.05 * range_ * np.random.randn(self.dim)
            x_trial += perturbation
            
            x_trial = np.clip(x_trial, lo, hi)
            return x_trial
        
        # 1. Initialize Population with LHS
        pop_size = 60
        if self.budget <= pop_size:
            pop_size = self.budget
        
        X = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[perm, i] = np.linspace(0, 1, pop_size)
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # 2. NSGA-II Evolutionary Loop
        while evals < self.budget and time_check():
            # Non-dominated sorting
            fronts = non_dominated_sort(F)
            
            # Create mating pool from first few fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    break
                mating_pool.extend(front)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and evals < self.budget and time_check():
                # Select parents from mating pool using tournament
                p1_idx = np.random.choice(mating_pool)
                p2_idx = np.random.choice(mating_pool)
                
                # Differential mutation
                x_child = differential_mutation(X, F, pop_size, lo, hi, range_, evals, problem, deadline)
                
                fc = problem(x_child)
                evals += 1
                
                offspring_X.append(x_child)
                offspring_F.append(fc)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        # Calculate crowding distance
                        cd = crowding_distance(combined_F, front)
                        # Sort by crowding distance descending
                        order = np.argsort(cd)[::-1]
                        for i in range(min(remaining, len(order))):
                            idx = front[order[i]]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break
            
            if len(next_X) > 0:
                X = np.array(next_X)
                F = np.array(next_F)
        
        # 3. Final polish: local search on non-dominated points
        if evals < self.budget and time_check():
            fronts = non_dominated_sort(F)
            nd_indices = fronts[0] if len(fronts) > 0 else []
            
            if len(nd_indices) > 0:
                # Polish top non-dominated points with local search
                n_polish = min(5, len(nd_indices))
                polish_budget = self.budget - evals
                evals_per_polish = max(1, polish_budget // n_polish)
                
                for idx in nd_indices[:n_polish]:
                    if evals >= self.budget or not time_check():
                        break
                    
                    x_curr = X[idx].copy()
                    f_curr = F[idx].copy()
                    best_x = x_curr.copy()
                    best_f = f_curr.copy()
                    
                    # Simple local search with adaptive step
                    step_size = 0.1 * range_
                    for _ in range(evals_per_polish):
                        if evals >= self.budget or not time_check():
                            break
                        
                        # Try random directions
                        for _ in range(2):
                            direction = np.random.randn(self.dim)
                            direction /= (np.linalg.norm(direction) + 1e-10)
                            x_trial = x_curr + step_size * direction
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = problem(x_trial)
                            evals += 1
                            
                            # Check if trial is non-dominated with respect to current best
                            if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                                best_x = x_trial
                                best_f = f_trial
                                x_curr = x_trial
                                step_size *= 0.7
                            elif np.all(best_f <= f_trial) and np.any(best_f < f_trial):
                                step_size *= 0.5
                            else:
                                # Non-dominated, keep as alternative
                                if np.random.rand() < 0.3:
                                    x_curr = x_trial
                                step_size *= 0.8
                    
                    X[idx] = best_x
                    F[idx] = best_f
        
        # 4. Return non-dominated set
        fronts = non_dominated_sort(F)
        nd_indices = fronts[0] if len(fronts) > 0 else list(range(len(F)))
        
        result_X = X[nd_indices]
        return result_X
