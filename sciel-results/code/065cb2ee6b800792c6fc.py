import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        # Use a fixed number of random samples to find a good basin
        n_explore = 80
        if budget <= n_explore:
            n_explore = max(1, budget // 2)
            n_refine = budget - n_explore
        else:
            n_refine = budget - n_explore
            
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Evaluate random points
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement using Nelder-Mead
        # We will run Nelder-Mead from the best point found.
        # If we have budget left, we might restart from the best point if it gets stuck, 
        # but for simplicity and robustness, we run one long Nelder-Mead.
        
        if best_x is None:
            # Fallback if no points were evaluated (should not happen)
            return (np.zeros(n), np.inf)
            
        # Initialize Simplex
        # The simplex is a set of n+1 points.
        # Start at best_x, and add small perturbations.
        # Scale of perturbation: 0.1 * (max - min) of domain? Or absolute?
        # Domain is [-5, 5], so width 10. 0.1 is 1.0 unit.
        step = 0.5
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(1, n + 1):
            simplex[i] = best_x
            simplex[i, i-1] += step
            
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        fvals[0] = best_f
        calls_used = n_explore
        
        for i in range(1, n + 1):
            if calls_used >= budget or time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls_used += 1
            
        # Nelder-Mead Iteration
        # Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        max_iter = 500 # Max iterations for NM
        
        for _ in range(max_iter):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Check convergence
            if fvals[-1] - fvals[0] < 1e-10:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if calls_used >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls_used += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if calls_used >= budget or time.time() > deadline:
                    break
                f_e = func(x_e)
                calls_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                if calls_used >= budget or time.time() > deadline:
                    break
                f_c = func(x_c)
                calls_used += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls_used >= budget or time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                    if time.time() > deadline:
                        break

        # Final Best
        # The best point is simplex[0] after sorting, but we might have lost track if we broke early.
        # We should track the global best explicitly.
        
        # Re-evaluate the current best candidate to be sure? No, we have the value.
        # However, the simplex might have moved. The best point in the current simplex is simplex[0].
        # But what if the best point was in the previous simplex and got replaced?
        # We should maintain a global best_x and best_f.
        
        # Let's re-implement the loop to track global best properly.
        
        # --- Revised Implementation ---
        
        # Reset for clean logic
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        
        n_explore = 80
        if budget <= n_explore:
            n_explore = max(1, budget // 2)
            n_refine = budget - n_explore
        else:
            n_refine = budget - n_explore
            
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            return (np.zeros(n), np.inf)
            
        # Initialize Simplex around best_x
        step = 0.5
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(1, n + 1):
            simplex[i] = best_x
            simplex[i, i-1] += step
            
        fvals = np.zeros(n + 1)
        fvals[0] = best_f
        calls_used = n_explore
        
        for i in range(1, n + 1):
            if calls_used >= budget or time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls_used += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
                
        # Nelder-Mead
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 1000
        
        for _ in range(max_iter):
            if calls_used >= budget or time.time() > deadline:
                break
            
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            if fvals[-1] - fvals[0] < 1e-10:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if calls_used >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls_used += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
                
            if f_r < fvals[0]:
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if calls_used >= budget or time.time() > deadline:
                    break
                f_e = func(x_e)
                calls_used += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                    
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                if calls_used >= budget or time.time() > deadline:
                    break
                f_c = func(x_c)
                calls_used += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                    
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls_used >= budget or time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    if time.time() > deadline:
                        break

        return best_x, best_f
