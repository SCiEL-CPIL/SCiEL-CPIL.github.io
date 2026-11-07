import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285.0
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def check_time():
            return time.time() < deadline and calls < budget
        
        # Phase 1: Dense local search around center (0,0,0,0,0)
        # Sharp ridge often has structure near center or along axes.
        # We do a quick radial search to find the general direction.
        
        # 1. Evaluate center
        center = np.zeros(n)
        if check_time():
            best_f = func(center)
            best_x = center.copy()
            calls += 1
        
        # 2. Evaluate points along axes (positive and negative) at various radii
        # This helps identify the "ridge" direction if it's axis-aligned or close to it.
        radii = [0.5, 1.0, 2.0, 3.0, 4.0]
        axis_points = []
        for r in radii:
            for i in range(n):
                for sign in [1, -1]:
                    p = np.zeros(n)
                    p[i] = sign * r
                    axis_points.append(p)
        
        # Shuffle to randomize order slightly, but keep structure
        # Actually, for AOCC, early progress is key. Let's just evaluate them in order of radius.
        for p in axis_points:
            if not check_time():
                break
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # Phase 2: Nelder-Mead starting from the best point found so far
        # Initialize simplex
        delta = 0.5
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(n):
            simplex[i + 1] = best_x
            simplex[i + 1, i] += delta
            
        # Clip to box
        simplex = np.clip(simplex, -5, 5)
        
        f_simplex = np.zeros(n + 1)
        
        # Evaluate initial simplex
        for i in range(n + 1):
            if not check_time():
                break
            f_simplex[i] = func(simplex[i])
            calls += 1
            if f_simplex[i] < best_f:
                best_f = f_simplex[i]
                best_x = simplex[i].copy()
        
        if not check_time():
            return best_x, best_f
            
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main Nelder-Mead loop
        while check_time():
            # Sort simplex
            idx = np.argsort(f_simplex)
            simplex_sorted = simplex[idx]
            f_sorted = f_simplex[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = simplex_sorted[0].copy()
                
            # Check convergence
            if np.max(f_sorted) - np.min(f_sorted) < 1e-12:
                break
                
            centroid = np.mean(simplex_sorted[:n], axis=0)
            worst = simplex_sorted[n]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            
            if not check_time():
                break
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_sorted[0]:
                # Expansion
                if not check_time():
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex_sorted[n] = x_e
                    f_sorted[n] = f_e
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                else:
                    simplex_sorted[n] = x_r
                    f_sorted[n] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
            elif f_r < f_sorted[n - 1]:
                # Accept reflection
                simplex_sorted[n] = x_r
                f_sorted[n] = f_r
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
            else:
                # Contraction
                if f_r < f_sorted[n]:
                    # Outside contraction
                    x_c = centroid + rho * (worst - centroid)
                else:
                    # Inside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                
                if not check_time():
                    break
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, f_sorted[n]):
                    simplex_sorted[n] = x_c
                    f_sorted[n] = f_c
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex_sorted[i] = simplex_sorted[0] + sigma * (simplex_sorted[i] - simplex_sorted[0])
                        simplex_sorted[i] = np.clip(simplex_sorted[i], -5, 5)
                        
                        if not check_time():
                            break
                        f_sorted[i] = func(simplex_sorted[i])
                        calls += 1
                        if f_sorted[i] < best_f:
                            best_f = f_sorted[i]
                            best_x = simplex_sorted[i].copy()
                            
                    if not check_time():
                        break
                    
            simplex = simplex_sorted
            f_simplex = f_sorted
            
        return best_x, best_f
