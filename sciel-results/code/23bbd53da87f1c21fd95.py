import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin to ensure we finish before the 330s hard kill
        deadline = start_time + 290.0
        
        # We need to track the global best and the number of calls
        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = self.budget
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility if needed, but random is fine
        
        def check_deadline():
            return time.time() > deadline - 5.0 or calls_used >= max_calls

        # Nelder-Mead implementation tailored for budget constraints
        def nelder_mead(x0, func, budget_left, current_best_f):
            nonlocal calls_used, best_x, best_f
            
            if budget_left < 10:
                return
            
            # Initialize simplex
            # x0 is the center
            # Create vertices: x0 and x0 + delta * e_i
            # Adaptive delta based on box size
            delta = 0.5 
            
            simplex = [x0.copy()]
            for i in range(self.dim):
                v = x0.copy()
                v[i] += delta
                # Ensure within bounds
                v = np.clip(v, -5, 5)
                simplex.append(v)
            
            # Evaluate initial simplex
            f_vals = []
            for v in simplex:
                if check_deadline():
                    return
                f = func(v)
                calls_used += 1
                f_vals.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = v.copy()
            
            # Sort simplex by function value
            # We store (f, x) pairs
            points = list(zip(f_vals, simplex))
            points.sort(key=lambda p: p[0])
            f_vals = [p[0] for p in points]
            simplex = [p[1] for p in points]
            
            # Main loop
            max_iter = 100 # Safety cap on iterations per NM run
            for _ in range(max_iter):
                if check_deadline():
                    return
                if budget_left < 10:
                    return

                # 1. Reflection
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if check_deadline():
                    return
                f_reflect = func(x_reflect)
                calls_used += 1
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()
                
                # 2. Expansion
                if f_reflect < f_vals[0]:
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    
                    if check_deadline():
                        return
                    f_expand = func(x_expand)
                    calls_used += 1
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                # 3. Contraction (Outside)
                elif f_reflect < f_vals[-2]:
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (x_worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    
                    if check_deadline():
                        return
                    f_contract = func(x_contract)
                    calls_used += 1
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = x_contract.copy()
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # 4. Shrink
                        # Move all points except the best towards the best
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if check_deadline():
                                return
                            f_shrink = func(simplex[i])
                            calls_used += 1
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                            f_vals[i] = f_shrink
                        
                        # Re-sort after shrink
                        points = list(zip(f_vals, simplex))
                        points.sort(key=lambda p: p[0])
                        f_vals = [p[0] for p in points]
                        simplex = [p[1] for p in points]
                        
                        # Check for convergence (optional, but good for stopping early)
                        if f_vals[0] > 1e-10:
                             if abs(f_vals[-1] - f_vals[0]) < 1e-8:
                                 break
                        else:
                             if abs(f_vals[-1] - f_vals[0]) < 1e-8:
                                 break

        # Strategy: Multi-Start Nelder-Mead
        # 1. Do an initial random search to find a decent starting point
        # 2. Run NM from there
        # 3. If time/calls remain, do a few random restarts and run NM from those too
        #    This helps escape local minima of the Gallagher function.
        
        # Phase 1: Initial Exploration
        # Use ~10% of budget for initial random search to find a good basin
        initial_budget = max(10, int(0.1 * max_calls))
        
        # First point
        x0 = rng.uniform(-5, 5, self.dim)
        f0 = func(x0)
        calls_used += 1
        best_x = x0.copy()
        best_f = f0
        
        # Quick random search for initial best
        for _ in range(initial_budget - 1):
            if check_deadline():
                break
            x_r = rng.uniform(-5, 5, self.dim)
            f_r = func(x_r)
            calls_used += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
        
        # Phase 2: Local Optimization from best found
        # Run NM from the global best
        nelder_mead(best_x, func, max_calls - calls_used, best_f)
        
        # Phase 3: Additional Restarts if budget allows
        # The Gallagher function has 21 peaks. We might be in a local minimum.
        # Let's try 2-3 more restarts if we have budget left.
        restarts = 0
        max_restarts = 3
        
        while restarts < max_restarts and not check_deadline():
            # Check if we have enough budget for a decent NM run (e.g., 50 calls)
            if max_calls - calls_used < 50:
                break
            
            # Random restart
            x_r = rng.uniform(-5, 5, self.dim)
            f_r = func(x_r)
            calls_used += 1
            
            # Only proceed if this restart point is not significantly worse than current best?
            # Or just always try?
            # Since NM is local, starting from a very bad point is a waste.
            # But if we are stuck in a deep local min, a bad start might lead to a better global min.
            # Let's just run NM from the new random point.
            
            # Run NM from this new point
            # We pass current best_f so we can update global best if we find something better
            # Note: NM will modify local state, but we track global best_x/best_f
            nelder_mead(x_r, func, max_calls - calls_used, best_f)
            
            restarts += 1
        
        return best_x, best_f
