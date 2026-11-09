import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val
        
        # Initial random sampling to get a starting point and diversity
        # We use a mix of random points and some structured points
        
        # Phase 1: Global Exploration with Adaptive Differential Evolution (JADE-like)
        # Parameters
        pop_size = 10  # Reasonable for dim=5 and budget=500
        max_generations = budget // pop_size  # ~50 generations
        
        # Initialize population
        # Start with a spread of points
        pop = np.zeros((pop_size, dim))
        fit = np.zeros(pop_size)
        
        # First few points: corners and center
        # Center
        pop[0] = np.zeros(dim)
        # Random points
        for i in range(1, pop_size):
            pop[i] = np.random.uniform(-5, 5, dim)
        
        # Evaluate initial population
        for i in range(pop_size):
            fit[i] = track_call(pop[i])
            
        # Sort initial population
        idx = np.argsort(fit)
        pop = pop[idx]
        fit = fit[idx]
        
        # JADE-style parameters
        F = 0.5
        CR = 0.5
        # Archive of successful vectors
        archive = []
        
        for gen in range(max_generations):
            if time.time() > deadline - 5.0 or calls_used >= budget:
                break
                
            # Generate offspring
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                    
                # Select base vector: current individual
                base = pop[i]
                
                # Select two distinct random indices different from i
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                
                # Differential evolution variant
                # DE/rand/1/bin
                # v = pop[r1] + F * (pop[r2] - pop[i])
                v = pop[r1] + F * (pop[r2] - base)
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = np.copy(base)
                # Random crossover
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Evaluate trial
                trial_fit = track_call(trial)
                
                # Selection
                if trial_fit <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    # Add to archive
                    archive.append(trial.copy())
                    if len(archive) > 10:
                        archive.pop(0)
                else:
                    # If failed, maybe use archive? (Simplified JADE)
                    pass
            
            # Sort population
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # Adapt F and CR (Simplified JADE adaptation)
            # In full JADE, we track successful F and CR values.
            # Here we use a simple heuristic: if progress is slow, increase exploration.
            # For now, keep F and CR fixed or slightly adaptive.
            # A common simple adaptation:
            # F = 0.5 + 0.1 * np.random.rand()
            # CR = 0.5 + 0.1 * np.random.rand()
            # But this might be too noisy. Let's stick to fixed for stability in low budget.
            
        # Phase 2: Local Refinement
        # Use remaining budget for local search around the best solution found
        if calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            if remaining > 5:
                # Use Nelder-Mead for local optimization
                from scipy.optimize import minimize
                
                # Define a wrapper that counts calls
                local_calls = 0
                def local_func(x):
                    nonlocal local_calls
                    if local_calls >= remaining:
                        return best_f
                    local_calls += 1
                    x_clipped = np.clip(x, -5, 5)
                    val = func(x_clipped)
                    calls_used += 1
                    if val < best_f:
                        best_f = val
                        best_x = x_clipped.copy()
                    return val
                
                # Initial simplex size
                # Nelder-Mead uses a simplex of size dim+1
                # We limit the number of iterations to stay within budget
                # Each iteration uses at most dim+1 evaluations, but usually less.
                # Let's be conservative: maxiter = remaining // (dim + 1)
                maxiter = max(1, remaining // (dim + 1))
                
                try:
                    res = minimize(local_func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass
        
        return best_x, best_f
