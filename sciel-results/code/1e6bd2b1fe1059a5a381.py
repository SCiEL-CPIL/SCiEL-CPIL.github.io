import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3000.0, 0.06])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        evals_used = 0
        max_evals = self.budget
        
        def safe_eval(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            if time.time() > deadline:
                return None
            try:
                f = problem(x)
                evals_used += 1
                return np.array(f)
            except:
                evals_used += 1
                return np.array([np.inf, np.inf])

        # 1. Initialization: Stratified Random Sampling
        # Use 20 points for a good initial spread
        init_size = min(20, max_evals)
        X = np.zeros((init_size, dim))
        F = np.zeros((init_size, self.n_obj))
        
        for i in range(init_size):
            x = lo + (hi - lo) * np.random.rand(dim)
            X[i] = x
            F[i] = safe_eval(x)
            
        if F is None or np.any(np.isnan(F)):
             return X

        # Function to compute non-dominated mask
        def get_non_dominated_mask(F):
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                is_dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd[i] = True
            return nd

        # Scalarized objective for local search
        # Normalize by reference point to balance scales
        def scalar_obj(x):
            f = safe_eval(x)
            if f is None or np.any(np.isnan(f)):
                return 1e10
            # If any objective is worse than reference, penalize heavily
            if np.any(f >= self.ref_point):
                return 1e10 + np.sum(f)
            # Minimize normalized sum
            return np.sum(f / self.ref_point)

        # Differential Evolution Phase
        # Remaining budget for DE and Local Search
        # Let's use 60% of remaining for DE, 40% for Local Polish
        remaining = max_evals - evals_used
        de_budget = int(remaining * 0.6)
        local_budget = remaining - de_budget
        
        pop_size = 10
        # Ensure we have enough budget for DE
        if de_budget >= pop_size * 2:
            # Keep current best non-dominated as part of population if they are good
            # But for simplicity, we run DE on the whole space, initialized with current best
            # We will maintain a population of size pop_size
            
            # Initialize DE population with current non-dominated points and randoms
            # If we have fewer than pop_size non-dominated, fill with randoms
            nd_mask = get_non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            de_pop_X = np.zeros((pop_size, dim))
            de_pop_F = np.zeros((pop_size, self.n_obj))
            
            # Fill with best non-dominated
            n_nd = min(pop_size, len(nd_indices))
            for i in range(n_nd):
                idx = nd_indices[i]
                de_pop_X[i] = X[idx]
                de_pop_F[i] = F[idx]
            
            # Fill rest with random
            for i in range(n_nd, pop_size):
                x = lo + (hi - lo) * np.random.rand(dim)
                de_pop_X[i] = x
                de_pop_F[i] = safe_eval(x)
                
            # DE Loop
            while evals_used < max_evals - local_budget: # Leave budget for local
                if time.time() > deadline:
                    break
                    
                # Select best, rand1, rand2
                # Sort by non-domination and crowding (approx by sum)
                # For simplicity, use sum of normalized objectives as fitness for selection
                fitness = np.sum(de_pop_F / self.ref_point, axis=1)
                # Handle inf/nan
                fitness[~np.isfinite(fitness)] = 1e10
                sorted_idx = np.argsort(fitness)
                
                best_idx = sorted_idx[0]
                rand1, rand2 = np.random.choice(pop_size, 2, replace=False)
                while rand1 == best_idx or rand2 == best_idx or rand1 == rand2:
                    rand1, rand2 = np.random.choice(pop_size, 2, replace=False)
                    
                # Mutation
                F_mut = de_pop_F[best_idx] + 0.5 * (de_pop_F[rand1] - de_pop_F[rand2])
                # Note: This is objective-space mutation, which is invalid for x-space.
                # We must mutate in x-space.
                
                # Correct DE: Mutate in X space
                x_mut = de_pop_X[best_idx] + 0.5 * (de_pop_X[rand1] - de_pop_X[rand2])
                x_mut = np.clip(x_mut, lo, hi)
                
                f_mut = safe_eval(x_mut)
                if f_mut is None:
                    break
                
                # Crossover
                j_rand = np.random.randint(dim)
                CR = 0.5
                x_trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        x_trial[j] = x_mut[j]
                    else:
                        x_trial[j] = de_pop_X[rand1][j] # Use rand1 as target for simplicity in this variant, or best?
                        # Standard DE: target is usually the i-th individual.
                        # Let's stick to a simple mutation of the best and replace worst.
                        
                # Simpler Strategy: Replace the worst individual in the current DE population with the mutated one if it's better.
                # This is a (1+1) EA on the DE population.
                
                # Let's switch to a simpler, more robust approach for small budgets:
                # Just do random mutation of the best non-dominated points.
                
                break # Break DE loop, switch to simpler local search strategy below
        
        # Simpler Strategy: Iterative Local Search from Best Points
        # We have used some evals. Let's just do local search from the current best non-dominated points.
        
        # Get current non-dominated
        nd_mask = get_non_dominated_mask(F)
        nd_indices = np.where(nd_mask)[0]
        
        # Sort non-dominated by scalar objective to pick best starting points
        nd_F = F[nd_indices]
        nd_X = X[nd_indices]
        
        # If no non-dominated found (shouldn't happen), use all
        if len(nd_X) == 0:
             nd_X = X
             nd_F = F

        # Select top K starting points for local search
        K = min(5, len(nd_X))
        
        # Sort by scalar objective
        scalar_scores = np.sum(nd_F / self.ref_point, axis=1)
        scalar_scores[~np.isfinite(scalar_scores)] = 1e10
        top_idx = np.argsort(scalar_scores)[:K]
        
        starts = nd_X[top_idx]
        
        # Local Search Loop
        while evals_used < max_evals:
            if time.time() > deadline:
                break
                
            # Pick a start point (cycle through them)
            start_idx = evals_used % len(starts)
            x0 = starts[start_idx]
            
            # Run Nelder-Mead
            res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1e9: # Valid result
                x_new = res.x
                # Clip to bounds
                x_new = np.clip(x_new, lo, hi)
                
                # Evaluate final point if not already evaluated (NM evaluates it)
                # But safe_eval inside scalar_obj might have hit the limit.
                # Let's explicitly evaluate to ensure it's recorded
                f_new = safe_eval(x_new)
                if f_new is not None:
                    # Add to population
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, f_new])
                    
                    # Update starts if this is better
                    s_new = np.sum(f_new / self.ref_point)
                    if s_new < scalar_scores[start_idx]:
                         starts[start_idx] = x_new
                         scalar_scores[start_idx] = s_new
                         nd_mask = get_non_dominated_mask(F)
                         # Re-calculate nd indices is expensive, just update the specific start
                         pass
                         
        return X
