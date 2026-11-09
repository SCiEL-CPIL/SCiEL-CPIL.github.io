"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f14 (Different powers).
    
    Strategy:
    1. Brief multi-restart (1+1)-ES for global exploration.
    2. Nelder-Mead simplex for local refinement.
    3. Time-aware: stops before wall-clock limit with safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget
        
        def check_time():
            nonlocal deadline
            return time.time() < deadline
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return None
            calls_made += 1
            return func(x)
        
        # Phase 1: Multi-restart (1+1)-ES for global exploration
        # Use ~40% of budget for exploration
        explore_budget = int(max_calls * 0.4)
        
        for restart in range(5):
            if calls_made >= max_calls or not check_time():
                break
            
            # Start from a random point
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 3.0
            steps_this_restart = explore_budget // 5
            
            for _ in range(steps_this_restart):
                if calls_made >= max_calls or not check_time():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
            
            # If we found a good point, do a quick local refinement
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead local refinement
        if best_x is not None and calls_made < max_calls and check_time():
            # Create initial simplex around best_x
            simplex = [best_x.copy()]
            for i in range(self.dim):
                perturbed = best_x.copy()
                perturbed[i] += 0.5
                perturbed[i] = np.clip(perturbed[i], -5, 5)
                simplex.append(perturbed)
            
            # Evaluate simplex
            simplex_f = []
            for point in simplex:
                if not check_time():
                    break
                fval = safe_call(point)
                if fval is None:
                    break
                simplex_f.append(fval)
            
            if len(simplex_f) == self.dim + 1:
                # Run Nelder-Mead
                alpha, gamma, rho, sigma_nm = 0.5, 2.0, 0.5, 0.5
                max_nm_steps = max_calls - calls_made
                
                for _ in range(max_nm_steps):
                    if not check_time() or calls_made >= max_calls:
                        break
                    
                    # Sort simplex by function value
                    idx = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in idx]
                    simplex_f = [simplex_f[i] for i in idx]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_call(xr)
                    if fr is None:
                        break
                    
                    if simplex_f[0] <= fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    elif fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[-1]:
                            # Outside contraction
                            xc = centroid + sigma_nm * (simplex[-1] - centroid)
                        else:
                            # Inside contraction
                            xc = centroid - sigma_nm * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        
                        if fc <= min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if not check_time():
                                    break
                                fval = safe_call(simplex[i])
                                if fval is None:
                                    simplex_f[i] = np.inf
                                else:
                                    simplex_f[i] = fval
                            break  # Stop Nelder-Mead after shrink
        
        # Phase 3: Final (1+1)-ES polish on best point
        if best_x is not None and calls_made < max_calls and check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = max_calls - calls_made
            
            for _ in range(remaining):
                if not check_time():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
