import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        initial_step = 0.5
        
        # Initialize Simplex
        # Start at origin, expand to create simplex
        simplex = [np.zeros(dim) for _ in range(dim + 1)]
        simplex_f = [0.0] * (dim + 1)
        
        # Evaluate initial points
        for i in range(dim + 1):
            if i > 0:
                simplex[i] = simplex[0].copy()
                simplex[i][i-1] = initial_step
            simplex_f[i] = safe_func(simplex[i])
            
            if calls >= budget or time.time() > deadline:
                return best_x, best_f
        
        # Main Loop
        max_iterations = budget
        iter_count = 0
        stall_counter = 0
        min_improvement = 1e-8
        last_best_f = best_f
        
        while calls < budget and time.time() < deadline and iter_count < max_iterations:
            iter_count += 1
            
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check for convergence/stagnation
            if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                # Restart from best point with smaller step
                new_simplex = [best_x.copy() for _ in range(dim + 1)]
                for i in range(1, dim + 1):
                    new_simplex[i][i-1] = initial_step * 0.5
                    new_simplex[i] = np.clip(new_simplex[i], -5, 5)
                
                for i in range(dim + 1):
                    new_simplex_f = safe_func(new_simplex[i])
                    if calls >= budget or time.time() > deadline:
                        return best_x, best_f
                simplex = new_simplex
                simplex_f = [safe_func(s) if False else 0 for s in simplex] # Re-eval not needed if we just evaluated, but let's keep it simple
                # Actually, we just evaluated them above. Let's re-eval to be safe with the loop structure
                for i in range(dim + 1):
                    simplex_f[i] = safe_func(simplex[i])
                    if calls >= budget or time.time() > deadline:
                        return best_x, best_f
                initial_step *= 0.5
                continue
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:dim], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[dim])
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_func(x_r)
            
            if calls >= budget or time.time() > deadline:
                return best_x, best_f
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_func(x_e)
                
                if calls >= budget or time.time() > deadline:
                    return best_x, best_f
                
                if f_e < f_r:
                    simplex[dim] = x_e
                    simplex_f[dim] = f_e
                else:
                    simplex[dim] = x_r
                    simplex_f[dim] = f_r
            elif f_r < simplex_f[dim-1]:
                # Accept Reflection
                simplex[dim] = x_r
                simplex_f[dim] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < worst, inside if f_r >= worst
                if f_r < simplex_f[dim]:
                    # Outside Contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside Contraction
                    x_c = centroid - 0.5 * (centroid - simplex[dim])
                
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_func(x_c)
                
                if calls >= budget or time.time() > deadline:
                    return best_x, best_f
                
                if f_c < min(f_r, simplex_f[dim]):
                    simplex[dim] = x_c
                    simplex_f[dim] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = safe_func(simplex[i])
                        
                        if calls >= budget or time.time() > deadline:
                            return best_x, best_f
                    
                    # Reset initial step if we shrunk too much
                    if initial_step > 1e-6:
                        initial_step *= 0.5
                    else:
                        # Restart with random perturbation
                        simplex[0] = best_x + np.random.normal(0, 0.1, dim)
                        simplex[0] = np.clip(simplex[0], -5, 5)
                        simplex_f[0] = safe_func(simplex[0])
                        if calls >= budget or time.time() > deadline:
                            return best_x, best_f
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0].copy()
                            simplex[i][i-1] = initial_step
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_func(simplex[i])
                            if calls >= budget or time.time() > deadline:
                                return best_x, best_f
        
        return best_x, best_f
