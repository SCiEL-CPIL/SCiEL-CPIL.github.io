import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_low = -5.0
        self.bound_high = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        x_best = None
        f_best = np.inf
        
        # Phase 1: Global Scan (70% of budget)
        # Use a stratified grid to cover the space [-5, 5]^5
        # 500 * 0.7 = 350 evaluations
        scan_budget = int(self.budget * 0.7)
        
        # A 4x4x4x4x4 grid = 1024 points, too many.
        # A 3x3x3x3x3 grid = 243 points. This fits well and provides good coverage.
        # We can add the center of the box as a 244th point if needed, or just stick to grid.
        # Let's use a slightly denser grid in the inner region or just a standard grid.
        # 243 points leaves 257 for local search.
        
        grid_points = []
        # 3 points per dimension: -4, 0, 4 (avoiding boundaries slightly to be safe, though func clips)
        # Actually, uniform spacing is better for global search.
        # np.linspace(-5, 5, 3) -> [-5, 0, 5]
        # Let's use 3 points per dim: -5, 0, 5
        # Total 3^5 = 243 points.
        
        coords = np.linspace(self.bound_low, self.bound_high, 3)
        # Generate all combinations
        for c0 in coords:
            for c1 in coords:
                for c2 in coords:
                    for c3 in coords:
                        for c4 in coords:
                            grid_points.append([c0, c1, c2, c3, c4])
        
        # Evaluate grid points
        # We need to track the best few basins to restart local search from.
        # Instead of evaluating all 243 first then picking, we can just evaluate all and keep the best.
        # But we need to save budget for local search.
        
        local_search_starts = []
        
        for p in grid_points:
            if time.time() > deadline - 10:
                break
            if len(local_search_starts) >= scan_budget:
                break
            x = np.array(p)
            try:
                fx = func(x)
            except:
                fx = np.inf
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            
            # Keep the best 5 points as candidates for local refinement
            # Simple approach: just keep the single best for now, or top 3.
            # Given the high multimodality, top 3 is better.
            # To keep it simple and fast, we'll just collect all valid points and sort later?
            # No, that requires storing 243 floats. That's fine.
            local_search_starts.append((fx, x.copy()))

        # Sort by fitness
        local_search_starts.sort(key=lambda item: item[0])
        
        # Select top 3 unique basins for local refinement
        # "Unique" is hard to define strictly, but distinct grid points are distinct.
        # We will take the top 3.
        candidates = [x for f, x in local_search_starts[:3]]
        
        # Phase 2: Local Refinement (30% of budget)
        # Use Nelder-Mead from each candidate.
        # Remaining budget: self.budget - (number of grid points evaluated)
        # We evaluated up to 243 points.
        # Let's say we have ~257 evaluations left.
        # Allocate ~80 evaluations per candidate.
        
        evals_used = len(local_search_starts)
        evals_left = self.budget - evals_used
        
        if evals_left <= 0:
            return x_best, f_best
            
        num_candidates = len(candidates)
        if num_candidates == 0:
            return x_best, f_best
            
        evals_per_candidate = max(1, int(evals_left / num_candidates))
        
        for x0 in candidates:
            if time.time() > deadline - 5:
                break
                
            # Nelder-Mead implementation
            # Simple implementation to avoid scipy overhead and dependency issues if any
            # Standard NM:
            # 1. Create simplex
            # 2. Loop:
            #    a. Sort vertices by fitness
            #    b. Reflect
            #    c. If not better, Expand or Contract
            #    d. If still not better, Shrink
            
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Initial simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            evals_used += 1
            
            for i in range(dim):
                xi = x0.copy()
                # Step size: 0.05 if x0[i] != 0 else 0.05
                step = 0.05 if x0[i] != 0 else 0.05
                xi[i] += step
                # Clip to bounds
                xi = np.clip(xi, self.bound_low, self.bound_high)
                simplex.append(xi)
                try:
                    fi = func(xi)
                except:
                    fi = np.inf
                f_simplex.append(fi)
                evals_used += 1
                
                if evals_used >= self.budget:
                    break
            
            if evals_used >= self.budget:
                break
            
            # Main loop
            max_iter = 50 # Hard cap on iterations to ensure we finish
            for _ in range(max_iter):
                if evals_used >= self.budget:
                    break
                if time.time() > deadline - 2:
                    break
                    
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, self.bound_low, self.bound_high)
                try:
                    f_r = func(x_r)
                except:
                    f_r = np.inf
                evals_used += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, self.bound_low, self.bound_high)
                    try:
                        f_e = func(x_e)
                    except:
                        f_e = np.inf
                    evals_used += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        
                elif f_r < f_simplex[-2]:
                    # Reflection is good
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < f_worst, else inside
                    if f_r < f_worst:
                        x_c = centroid + rho * (centroid - worst)
                    else:
                        x_c = centroid - rho * (centroid - worst)
                    
                    x_c = np.clip(x_c, self.bound_low, self.bound_high)
                    try:
                        f_c = func(x_c)
                    except:
                        f_c = np.inf
                    evals_used += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], self.bound_low, self.bound_high)
                            try:
                                f_simplex[i] = func(simplex[i])
                            except:
                                f_simplex[i] = np.inf
                            evals_used += 1
                            
                            if evals_used >= self.budget:
                                break
                        
                        if evals_used >= self.budget:
                            break
                
                # Update global best
                for i in range(len(simplex)):
                    if f_simplex[i] < f_best:
                        f_best = f_simplex[i]
                        x_best = simplex[i].copy()

        return x_best, f_best
