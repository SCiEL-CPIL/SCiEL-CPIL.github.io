import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sparse global lattice search to identify basin of attraction
        # Use a coarse grid in [-4,4]^5 (where minimizer is known to lie)
        # This provides better global coverage than random sampling for finding
        # the basin of the rotated Rosenbrock function
        
        # Evaluate origin first
        tracked_func(np.zeros(self.dim))
        
        # Sparse lattice: use 3 points per dimension in [-4, 4]
        # This gives 3^5 = 243 points, but we'll subsample to stay within budget
        # Use 2 points per dimension initially: 2^5 = 32 points
        lattice_points = []
        for i in range(2 ** self.dim):
            point = np.zeros(self.dim)
            for d in range(self.dim):
                if (i >> d) & 1:
                    point[d] = 4.0
                else:
                    point[d] = -4.0
            lattice_points.append(point)
        
        # Also add midpoints for finer coverage: 4^5 = 1024, but we'll use subset
        # Add the center of each edge of the hypercube
        midpoints = []
        for d in range(self.dim):
            for val in [-2.0, 0.0, 2.0]:
                point = np.zeros(self.dim)
                point[d] = val
                midpoints.append(point)
        
        # Combine and evaluate, staying within budget
        all_starts = lattice_points + midpoints + [np.zeros(self.dim)]
        
        # Shuffle for diversity
        rng.shuffle(all_starts)
        
        for x0 in all_starts:
            if call_count >= self.budget - 30:
                break
            tracked_func(x0)
        
        # Phase 2: Nelder-Mead simplex optimization from best point
        # Nelder-Mead exploits smoothness without gradient noise
        if time_ok() and call_count < self.budget - 5:
            try:
                remaining = self.budget - call_count - 5
                if remaining > 10:
                    # Initialize simplex around best_x
                    simplex = [best_x.copy()]
                    for d in range(self.dim):
                        x_delta = best_x.copy()
                        x_delta[d] += 0.1
                        x_delta = np.clip(x_delta, -5.0, 5.0)
                        simplex.append(x_delta)
                    
                    # Evaluate initial simplex
                    for i, x in enumerate(simplex):
                        if call_count >= self.budget - 5:
                            break
                        fx = tracked_func(x)
                        simplex[i] = (fx, x)
                    
                    # Sort simplex by function value
                    simplex.sort(key=lambda t: t[0])
                    
                    # Nelder-Mead iterations
                    alpha = 1.0   # reflection
                    gamma = 2.0   # expansion
                    rho = 0.5     # contraction
                    sigma = 0.5   # shrinkage
                    
                    for iteration in range(200):
                        if call_count >= self.budget - 5:
                            break
                        if not time_ok():
                            break
                        
                        # Get worst, second-worst, and best points
                        x_worst = simplex[-1][1]
                        fx_worst = simplex[-1][0]
                        x_second_worst = simplex[-2][1]
                        fx_second_worst = simplex[-2][0]
                        x_best = simplex[0][1]
                        fx_best = simplex[0][0]
                        
                        # Compute centroid of all but worst point
                        centroid = np.mean([s[1] for s in simplex[:-1]], axis=0)
                        
                        # Reflection
                        x_reflect = centroid + alpha * (centroid - x_worst)
                        x_reflect = np.clip(x_reflect, -5.0, 5.0)
                        fx_reflect = tracked_func(x_reflect)
                        
                        if call_count >= self.budget - 5:
                            break
                        if not time_ok():
                            break
                        
                        if fx_reflect < fx_best:
                            # Try expansion
                            x_expand = centroid + gamma * (x_reflect - centroid)
                            x_expand = np.clip(x_expand, -5.0, 5.0)
                            fx_expand = tracked_func(x_expand)
                            
                            if call_count >= self.budget - 5:
                                break
                            if not time_ok():
                                break
                            
                            if fx_expand < fx_reflect:
                                simplex[-1] = (fx_expand, x_expand)
                            else:
                                simplex[-1] = (fx_reflect, x_reflect)
                        elif fx_reflect < fx_second_worst:
                            # Accept reflection
                            simplex[-1] = (fx_reflect, x_reflect)
                        else:
                            # Contraction
                            if fx_reflect < fx_worst:
                                # Outside contraction
                                x_contract = centroid + rho * (x_reflect - centroid)
                            else:
                                # Inside contraction
                                x_contract = centroid - rho * (x_worst - centroid)
                            
                            x_contract = np.clip(x_contract, -5.0, 5.0)
                            fx_contract = tracked_func(x_contract)
                            
                            if call_count >= self.budget - 5:
                                break
                            if not time_ok():
                                break
                            
                            if fx_contract < min(fx_reflect, fx_worst):
                                simplex[-1] = (fx_contract, x_contract)
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    x_shrink = x_best + sigma * (simplex[i][1] - x_best)
                                    x_shrink = np.clip(x_shrink, -5.0, 5.0)
                                    fx_shrink = tracked_func(x_shrink)
                                    
                                    if call_count >= self.budget - 5:
                                        break
                                    if not time_ok():
                                        break
                                    
                                    simplex[i] = (fx_shrink, x_shrink)
                                break
                        
                        # Sort simplex
                        simplex.sort(key=lambda t: t[0])
                        
                        # Check convergence
                        if abs(fx_worst - fx_best) < 1e-10:
                            break
                    
                    # Update best from simplex
                    best_in_simplex = min(simplex, key=lambda t: t[0])
                    if best_in_simplex[0] < best_fx:
                        best_fx = best_in_simplex[0]
                        best_x = best_in_simplex[1].copy()
                        
            except Exception:
                pass
        
        # Phase 3: If budget remains, try a second Nelder-Mead from a different start
        if time_ok() and call_count < self.budget - 20:
            try:
                remaining = self.budget - call_count - 5
                if remaining > 20:
                    # Try from second-best point from lattice search
                    # Find a different starting point
                    alt_starts = [p for p in all_starts if not np.allclose(p, best_x)]
                    if alt_starts:
                        # Pick a diverse start
                        alt_x = alt_starts[0]
                        
                        # Quick Nelder-Mead from alt_x
                        simplex2 = [alt_x.copy()]
                        for d in range(self.dim):
                            x_delta = alt_x.copy()
                            x_delta[d] += 0.1
                            x_delta = np.clip(x_delta, -5.0, 5.0)
                            simplex2.append(x_delta)
                        
                        for i, x in enumerate(simplex2):
                            if call_count >= self.budget - 5:
                                break
                            fx = tracked_func(x)
                            simplex2[i] = (fx, x)
                        
                        simplex2.sort(key=lambda t: t[0])
                        
                        # Fewer iterations for this second attempt
                        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                        for iteration in range(50):
                            if call_count >= self.budget - 5:
                                break
                            if not time_ok():
                                break
                            
                            x_worst = simplex2[-1][1]
                            fx_worst = simplex2[-1][0]
                            x_second_worst = simplex2[-2][1]
                            fx_second_worst = simplex2[-2][0]
                            x_best = simplex2[0][1]
                            fx_best = simplex2[0][0]
                            
                            centroid = np.mean([s[1] for s in simplex2[:-1]], axis=0)
                            
                            x_reflect = centroid + alpha * (centroid - x_worst)
                            x_reflect = np.clip(x_reflect, -5.0, 5.0)
                            fx_reflect = tracked_func(x_reflect)
                            
                            if call_count >= self.budget - 5:
                                break
                            if not time_ok():
                                break
                            
                            if fx_reflect < fx_best:
                                x_expand = centroid + gamma * (x_reflect - centroid)
                                x_expand = np.clip(x_expand, -5.0, 5.0)
                                fx_expand = tracked_func(x_expand)
                                
                                if call_count >= self.budget - 5:
                                    break
                                if not time_ok():
                                    break
                                
                                if fx_expand < fx_reflect:
                                    simplex2[-1] = (fx_expand, x_expand)
                                else:
                                    simplex2[-1] = (fx_reflect, x_reflect)
                            elif fx_reflect < fx_second_worst:
                                simplex2[-1] = (fx_reflect, x_reflect)
                            else:
                                if fx_reflect < fx_worst:
                                    x_contract = centroid + rho * (x_reflect - centroid)
                                else:
                                    x_contract = centroid - rho * (x_worst - centroid)
                                
                                x_contract = np.clip(x_contract, -5.0, 5.0)
                                fx_contract = tracked_func(x_contract)
                                
                                if call_count >= self.budget - 5:
                                    break
                                if not time_ok():
                                    break
                                
                                if fx_contract < min(fx_reflect, fx_worst):
                                    simplex2[-1] = (fx_contract, x_contract)
                                else:
                                    for i in range(1, len(simplex2)):
                                        x_shrink = x_best + sigma * (simplex2[i][1] - x_best)
                                        x_shrink = np.clip(x_shrink, -5.0, 5.0)
                                        fx_shrink = tracked_func(x_shrink)
                                        
                                        if call_count >= self.budget - 5:
                                            break
                                        if not time_ok():
                                            break
                                        
                                        simplex2[i] = (fx_shrink, x_shrink)
                                    break
                            
                            simplex2.sort(key=lambda t: t[0])
                            
                            if abs(fx_worst - fx_best) < 1e-10:
                                break
                        
                        best_in_simplex2 = min(simplex2, key=lambda t: t[0])
                        if best_in_simplex2[0] < best_fx:
                            best_fx = best_in_simplex2[0]
                            best_x = best_in_simplex2[1].copy()
                            
            except Exception:
                pass

        return best_x, best_fx
