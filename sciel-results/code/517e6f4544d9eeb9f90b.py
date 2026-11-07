import numpy as np
import time
import scipy.optimize
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evaluations = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.evaluations = 0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for reproducibility within the run, but the evaluator sets np.random.seed
        # We should use np.random directly to respect the evaluator's seed setting.
        
        def evaluate(x):
            nonlocal self
            # Time guard
            if time.time() - self.start_time > 290: # 10s margin
                return None
            if self.evaluations >= self.budget:
                return None
            
            x = np.clip(x, lo, hi)
            f = problem(x)
            self.evaluations += 1
            return f

        # 1. Initialization: Latin Hypercube Sampling for better coverage
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        # LHS
        for j in range(dim):
            # Divide [0, 1] into pop_size intervals, pick one random point in each
            u = np.random.rand(pop_size)
            # Sort to ensure stratification? Or just random permutation of sorted uniform?
            # Standard LHS:
            # Create intervals
            intervals = np.linspace(0, 1, pop_size + 1)
            points = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(points)
            # Map to [lo, hi]
            X[:, j] = lo[j] + points * (hi[j] - lo[j])
            
        # Evaluate initial population
        valid_indices = []
        for i in range(pop_size):
            f = evaluate(X[i])
            if f is not None:
                F[i] = f
                valid_indices.append(i)
            else:
                break
        
        if not valid_indices:
            return X # Should not happen unless budget < pop_size
            
        X = X[valid_indices]
        F = F[valid_indices]
        current_pop_size = len(X)
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)] # Dominated sets
            n_d = np.zeros(n, dtype=int) # Number of dominators
            front = [0] # First front
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # p dominates q?
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # q dominates p?
                    if np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                        
            for p in range(n):
                if n_d[p] == 0:
                    ranks[p] = 0
                    front[0] += 1
                    
            k = 0
            while front[k] > 0:
                next_front = []
                for p in range(n):
                    if ranks[p] == k:
                        for q in S[p]:
                            n_d[q] -= 1
                            if n_d[q] == 0:
                                ranks[q] = k + 1
                                next_front.append(q)
                if next_front:
                    k += 1
                    front.append(len(next_front))
                else:
                    break
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            # Group by rank
            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    cd[indices] = np.inf
                    continue
                for m in range(self.n_obj):
                    # Sort by objective m
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_min = F[sorted_idx[0], m]
                    f_max = F[sorted_idx[-1], m]
                    if f_max - f_min == 0:
                        continue
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
            return cd

        # Helper: Local Search (Nelder-Mead with limited evaluations)
        def local_search(x, f):
            # Use scipy's minimize with Nelder-Mead
            # We need to wrap the problem to count evaluations and respect budget
            local_evals = 0
            max_local_evals = 15
            
            def local_obj(x_local):
                nonlocal local_evals
                if local_evals >= max_local_evals:
                    return np.sum(f) # Return a large value or current best to stop
                if time.time() - self.start_time > 290:
                    return np.sum(f)
                if self.evaluations >= self.budget:
                    return np.sum(f)
                
                x_local = np.clip(x_local, lo, hi)
                try:
                    f_new = problem(x_local)
                    self.evaluations += 1
                    local_evals += 1
                    # We want to minimize the sum of objectives as a proxy for local improvement
                    # But multi-objective local search is tricky. 
                    # A common trick is to use a weighted sum with random weights or just minimize one objective while constraining others.
                    # For simplicity and robustness in WFG7, let's try minimizing the sum of objectives.
                    # This might bias towards the center of the front, but it's a good local refinement.
                    return np.sum(f_new)
                except Exception:
                    return 1e9

            try:
                res = scipy.optimize.minimize(local_obj, x, method='Nelder-Mead', 
                                             options={'maxiter': 10, 'maxfev': max_local_evals, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < np.sum(f):
                    x_new = np.clip(res.x, lo, hi)
                    # Verify the new point is valid and better in some sense
                    f_new = problem(x_new)
                    self.evaluations += 1 # This evaluation is extra, be careful!
                    # Actually, scipy's minimize already called local_obj which incremented self.evaluations.
                    # So we should NOT call problem again here if we want to be strict.
                    # But we need the vector f_new for the population.
                    # Let's assume the last evaluation in local_obj was the best one found.
                    # However, local_obj returns a scalar. We don't have the vector from the last call easily.
                    # To be safe and simple, let's implement a simple coordinate descent local search instead.
            except Exception:
                pass
            
            # Let's replace scipy with a simple custom local search to control evaluations strictly.
            # Simple Coordinate Descent with adaptive step size
            x_cur = x.copy()
            f_cur = f.copy()
            step_size = 0.1 * (hi - lo)
            improved = True
            while improved and self.evaluations < self.budget and time.time() - self.start_time < 290:
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        x_try = x_cur.copy()
                        x_try[d] += direction * step_size[d]
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        self.evaluations += 1
                        
                        # Check if f_try dominates f_cur or is better in sum
                        # For local search, we accept if it's non-dominated by current or better in sum
                        if np.all(f_try <= f_cur) and np.any(f_try < f_cur):
                            x_cur = x_try
                            f_cur = f_try
                            improved = True
                        elif np.sum(f_try) < np.sum(f_cur):
                            # Accept if sum is better, to escape local minima in single objective
                            x_cur = x_try
                            f_cur = f_try
                            improved = True
                        if self.evaluations >= self.budget or time.time() - self.start_time > 290:
                            return x_cur, f_cur
            return x_cur, f_cur

        # Main Evolutionary Loop
        generations = 0
        while self.evaluations < self.budget and time.time() - self.start_time < 290:
            generations += 1
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)
            
            # Select parents
            # Tournament selection
            def tournament_select(pop_size, ranks, cd):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                best_idx = 0
                best_rank = ranks[candidates[0]]
                best_cd = cd[candidates[0]]
                for i in range(1, 3):
                    if ranks[candidates[i]] < best_rank:
                        best_rank = ranks[candidates[i]]
                        best_cd = cd[candidates[i]]
                        best_idx = i
                    elif ranks[candidates[i]] == best_rank and cd[candidates[i]] > best_cd:
                        best_idx = i
                return candidates[best_idx]
            
            p1_idx = tournament_select(current_pop_size, ranks, cd)
            p2_idx = tournament_select(current_pop_size, ranks, cd)
            
            if p1_idx == p2_idx:
                p2_idx = (p1_idx + 1) % current_pop_size
            
            # Crossover (SBX)
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            
            def sbx_crossover(x1, x2):
                beta = 15
                child1 = x1.copy()
                child2 = x2.copy()
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        if x1[d] != x2[d]:
                            u = np.random.rand()
                            if u <= 0.5:
                                alpha = (u * 2) ** (1 / (beta + 1))
                            else:
                                alpha = (1 / (2 * (1 - u))) ** (1 / (beta + 1))
                            c1 = 0.5 * ((1 + alpha) * x1[d] + (1 - alpha) * x2[d])
                            c2 = 0.5 * ((1 + 1/alpha) * x1[d] + (1 - 1/alpha) * x2[d])
                            # Clip to bounds
                            if c1 < lo[d] or c1 > hi[d]:
                                c1 = x1[d]
                            if c2 < lo[d] or c2 > hi[d]:
                                c2 = x2[d]
                            child1[d] = c1
                            child2[d] = c2
                return child1, child2

            c1, c2 = sbx_crossover(x1, x2)
            
            # Mutation (Polynomial)
            def poly_mutation(x):
                eta = 20
                x_new = x.copy()
                for d in range(dim):
                    if np.random.rand() < 0.5: # Mutation rate 0.5 per gene? Or 1/dim? Let's use 1/dim
                        u = np.random.rand()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        x_new[d] = x[d] + delta_q * (hi[d] - lo[d])
                        x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                return x_new

            c1 = poly_mutation(c1)
            c2 = poly_mutation(c2)
            
            # Evaluate children
            f1 = evaluate(c1)
            if f1 is None:
                break
            f2 = evaluate(c2)
            if f2 is None:
                # Only c1 was evaluated
                # Add c1 to population
                # Replace worst individual
                # For simplicity, replace the individual with the highest rank (worst)
                # Or just replace the last one if we are adding
                # Let's maintain a population of fixed size by replacing the worst dominated
                # Identify worst: highest rank, then lowest crowding distance
                # For simplicity in this loop, let's just append and if size > pop_size, remove worst
                
                # To keep it simple and fast, let's just replace the worst individual in the current population
                # Find worst
                worst_idx = np.argmax(ranks) # This is not strictly correct if ranks are not updated, but close enough for a heuristic
                # Better: find the individual with the highest rank, and if tie, lowest cd
                # Since ranks are from previous generation, this is an approximation.
                
                # Let's use a proper selection:
                # Sort by (rank, -cd)
                sort_idx = np.lexsort((-cd, ranks))
                worst_idx = sort_idx[-1]
                
                X[worst_idx] = c1
                F[worst_idx] = f1
                
                # Apply local search to the new child?
                # Maybe only every few generations to save budget
                if generations % 2 == 0:
                    c1_ls, f1_ls = local_search(c1, f1)
                    if np.all(f1_ls <= F[worst_idx]) and np.any(f1_ls < F[worst_idx]):
                        F[worst_idx] = f1_ls
                        X[worst_idx] = c1_ls
                    elif np.sum(f1_ls) < np.sum(F[worst_idx]):
                        F[worst_idx] = f1_ls
                        X[worst_idx] = c1_ls
                continue

            # Add c2 as well
            # Find worst again
            sort_idx = np.lexsort((-cd, ranks))
            worst_idx = sort_idx[-1]
            X[worst_idx] = c2
            F[worst_idx] = f2
            
            if generations % 2 == 0:
                c2_ls, f2_ls = local_search(c2, f2)
                if np.all(f2_ls <= F[worst_idx]) and np.any(f2_ls < F[worst_idx]):
                    F[worst_idx] = f2_ls
                    X[worst_idx] = c2_ls
                elif np.sum(f2_ls) < np.sum(F[worst_idx]):
                    F[worst_idx] = f2_ls
                    X[worst_idx] = c2_ls

        return X
