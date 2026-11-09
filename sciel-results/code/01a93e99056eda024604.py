import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Smart initialization - try several random points to find a good starting region
        num_initial = 5
        candidates = []
        for _ in range(num_initial):
            if time.time() > deadline or calls >= budget:
                break
            x_init = rng.uniform(-4.0, 4.0, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
        
        if not candidates:
            return best_x, best_f
        
        # Pick the best initial point
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        # Phase 2: Nelder-Mead with adaptive step size
        # Initial simplex around x0
        delta = 1.0
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += delta
            x_i = np.clip(x_i, -5.0, 5.0)
            f_i = safe_func(x_i)
            simplex.append(x_i)
            simplex_f.append(f_i)
            if time.time() > deadline or calls >= budget:
                return best_x, best_f
        
        # Nelder-Mead parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        max_iterations = 300
        iter_count = 0
        stall_counter = 0
        last_best_f = best_f
        min_improvement = 1e-10
        
        while calls < budget and time.time() < deadline and iter_count < max_iterations:
            iter_count += 1
            
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check for convergence/stagnation
            spread = simplex_f[-1] - simplex_f[0]
            if spread < 1e-12:
                # Restart from best point with smaller step
                delta *= 0.5
                if delta < 1e-6:
                    # Try random perturbation
                    x0 = best_x + rng.normal(0, 0.1, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    f0 = safe_func(x0)
                    simplex = [x0.copy()]
                    simplex_f = [f0]
                    for i in range(dim):
                        x_i = x0.copy()
                        x_i[i] += delta
                        x_i = np.clip(x_i, -5.0, 5.0)
                        f_i = safe_func(x_i)
                        simplex.append(x_i)
                        simplex_f.append(f_i)
                        if time.time() > deadline or calls >= budget:
                            return best_x, best_f
                    continue
                
                new_simplex = [best_x.copy() for _ in range(dim + 1)]
                for i in range(1, dim + 1):
                    new_simplex[i][i-1] = delta
                    new_simplex[i] = np.clip(new_simplex[i], -5.0, 5.0)
                
                new_simplex_f = []
                for i in range(dim + 1):
                    f_new = safe_func(new_simplex[i])
                    new_simplex_f.append(f_new)
                    if time.time() > deadline or calls >= budget:
                        return best_x, best_f
                
                simplex = new_simplex
                simplex_f = new_simplex_f
                continue
            
            # Check for stagnation
            if abs(best_f - last_best_f) < min_improvement:
                stall_counter += 1
            else:
                stall_counter = 0
                last_best_f = best_f
            
            # If stalled too long, restart with smaller step
            if stall_counter > 20:
                delta *= 0.5
                if delta < 1e-6:
                    x0 = best_x + rng.normal(0, 0.05, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    f0 = safe_func(x0)
                    simplex = [x0.copy()]
                    simplex_f = [f0]
                    for i in range(dim):
                        x_i = x0.copy()
                        x_i[i] += delta
                        x_i = np.clip(x_i, -5.0, 5.0)
                        f_i = safe_func(x_i)
                        simplex.append(x_i)
                        simplex_f.append(f_i)
                        if time.time() > deadline or calls >= budget:
                            return best_x, best_f
                    stall_counter = 0
                    continue
            
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:dim], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[dim])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = safe_func(x_r)
            
            if time.time() > deadline or calls >= budget:
                return best_x, best_f
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = safe_func(x_e)
                
                if time.time() > deadline or calls >= budget:
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
                if f_r < simplex_f[dim]:
                    # Outside Contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside Contraction
                    x_c = centroid - 0.5 * (centroid - simplex[dim])
                
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = safe_func(x_c)
                
                if time.time() > deadline or calls >= budget:
                    return best_x, best_f
                
                if f_c < min(f_r, simplex_f[dim]):
                    simplex[dim] = x_c
                    simplex_f[dim] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = safe_func(simplex[i])
                        
                        if time.time() > deadline or calls >= budget:
                            return best_x, best_f
                    
                    # Adapt step size based on simplex spread
                    if spread < 0.1:
                        delta *= 0.7
                    elif spread > 1.0:
                        delta *= 1.2
                    delta = np.clip(delta, 1e-6, 5.0)
        
        return best_x, best_f
