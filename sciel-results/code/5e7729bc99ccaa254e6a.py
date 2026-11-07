import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # 60s margin before 330s kill
        n = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        calls = 0
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x_clip = np.clip(x, -5, 5)
            val = func(x_clip)
            calls += 1
            return val

        # Nelder-Mead with multi-start
        # For Lunacek bi-Rastrigin, we need to handle the non-smooth Rastrigin component
        # Multi-start helps escape secondary basins
        
        rng = np.random.default_rng(42)
        
        # Strategy: Multiple Nelder-Mead restarts with different initializations
        # Each restart gets a fraction of the budget
        # Focus on early progress for AOCC
        
        # Phase 1: Coarse exploration with several short Nelder-Mead runs
        # Phase 2: Fine-tuning with the best result
        
        num_starts = 8
        budget_per_start = max(30, budget // num_starts)
        
        # Initial points: mix of random and structured
        init_points = []
        # Center
        init_points.append(np.zeros(n))
        # Random points
        for _ in range(num_starts - 1):
            init_points.append(rng.uniform(-4, 4, n))
        
        # Shuffle
        rng.shuffle(init_points)
        
        for start_idx, x0 in enumerate(init_points):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Check if we should stop early for this start
            remaining = budget - calls
            if remaining < 15:
                break
            
            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            tol = 1e-8
            max_iter = min(remaining, budget_per_start)
            
            # Build initial simplex
            simplex = [x0.copy()]
            f_vals = [safe_call(x0)]
            
            # Expand simplex
            for i in range(n):
                new_x = x0.copy()
                # Initial step size
                step = 0.5 if i == 0 else 0.3
                new_x[i] += step
                simplex.append(new_x)
                f_vals.append(safe_call(new_x))
            
            if calls >= budget:
                break
            
            # Nelder-Mead iteration
            for it in range(max_iter):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort by function value
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
                
                # Check convergence
                if abs(f_vals[0] - f_vals[-1]) < tol * (1 + abs(f_vals[0])):
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = safe_call(x_ref)
                
                if calls >= budget:
                    break
                
                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = safe_call(x_exp)
                    
                    if calls >= budget:
                        break
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    
                    f_con = safe_call(x_con)
                    
                    if calls >= budget:
                        break
                    
                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = safe_call(simplex[i])
                        
                        if calls >= budget:
                            break
                
                # Update global best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
            
            # Update global best after each start
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = simplex[0].copy()
        
        # If we have budget left, do a final refinement from the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining >= 10:
                # Final Nelder-Mead refinement
                x0 = best_x.copy()
                
                simplex = [x0.copy()]
                f_vals = [best_f]
                
                for i in range(n):
                    new_x = x0.copy()
                    step = 0.1
                    new_x[i] += step
                    simplex.append(new_x)
                    f_vals.append(safe_call(new_x))
                
                if calls < budget:
                    alpha = 1.0
                    gamma = 2.0
                    rho = 0.5
                    sigma = 0.5
                    tol = 1e-10
                    max_iter = min(remaining, 40)
                    
                    for it in range(max_iter):
                        if calls >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        idx = np.argsort(f_vals)
                        simplex = [simplex[i] for i in idx]
                        f_vals = [f_vals[i] for i in idx]
                        
                        if abs(f_vals[0] - f_vals[-1]) < tol * (1 + abs(f_vals[0])):
                            break
                        
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        x_ref = centroid + alpha * (centroid - simplex[-1])
                        f_ref = safe_call(x_ref)
                        
                        if calls >= budget:
                            break
                        
                        if f_ref < f_vals[0]:
                            x_exp = centroid + gamma * (centroid - simplex[-1])
                            f_exp = safe_call(x_exp)
                            
                            if calls >= budget:
                                break
                            
                            if f_exp < f_ref:
                                simplex[-1] = x_exp
                                f_vals[-1] = f_exp
                            else:
                                simplex[-1] = x_ref
                                f_vals[-1] = f_ref
                        elif f_ref < f_vals[-2]:
                            simplex[-1] = x_ref
                            f_vals[-1] = f_ref
                        else:
                            if f_ref < f_vals[-1]:
                                x_con = centroid + rho * (x_ref - centroid)
                            else:
                                x_con = centroid - rho * (centroid - simplex[-1])
                            
                            f_con = safe_call(x_con)
                            
                            if calls >= budget:
                                break
                            
                            if f_con < min(f_ref, f_vals[-1]):
                                simplex[-1] = x_con
                                f_vals[-1] = f_con
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    f_vals[i] = safe_call(simplex[i])
                                
                                if calls >= budget:
                                    break
                        
                        if f_vals[0] < best_f:
                            best_f = f_vals[0]
                            best_x = simplex[0].copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
        
        return best_x, best_f
