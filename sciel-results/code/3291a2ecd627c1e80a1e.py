"""
Optimizer using Multi-Start Nelder-Mead with Adaptive Coordinate Polish for BBOB f20 (Schwefel).
Structural refinement: Tightens coordinate polish, adds adaptive step sizing in NM,
and increases polish frequency to improve early-progress AOCC scoring.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Track global best
        best_x = np.zeros(n)
        best_fx = float('inf')
        calls = 0
        
        # Multi-start strategy: use restarts to explore different regions
        max_restarts = 12
        restart_budget_min = 30  # minimum calls per restart
        
        def coordinate_polish(x, f, budget_limit, max_steps=3):
            """
            Perform lightweight coordinate-wise line search to refine the point x.
            Uses fewer evaluations than before for faster convergence.
            """
            nonlocal calls
            x = x.copy()
            f = f
            # Adaptive step sizes: start moderate, shrink if no improvement
            base_steps = [0.8, 0.4, 0.2]
            
            for pass_num in range(max_steps):
                if calls >= budget_limit or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls >= budget_limit or time.time() > deadline:
                        break
                    best_local_f = f
                    best_local_x = x.copy()
                    
                    for step in base_steps:
                        if calls >= budget_limit or time.time() > deadline:
                            break
                        x_plus = x.copy()
                        x_plus[i] += step
                        x_minus = x.copy()
                        x_minus[i] -= step
                        
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = np.clip(x_minus, -5, 5)
                        
                        f_plus = func(x_plus)
                        calls += 1
                        if f_plus < best_local_f:
                            best_local_f = f_plus
                            best_local_x = x_plus.copy()
                            improved = True
                        
                        if calls >= budget_limit or time.time() > deadline:
                            break
                        f_minus = func(x_minus)
                        calls += 1
                        if f_minus < best_local_f:
                            best_local_f = f_minus
                            best_local_x = x_minus.copy()
                            improved = True
                    
                    if best_local_x is not x:
                        x = best_local_x
                        f = best_local_f
                        if f < best_fx:
                            best_fx = f
                            best_x = x.copy()
                
                if not improved:
                    break
            
            return x, f

        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Check if we have enough budget for a meaningful restart
            remaining = budget - calls
            if remaining < restart_budget_min:
                break
            
            # For first restart, start at origin; for subsequent, use diverse starts
            if restart == 0:
                x_start = np.zeros(n)
            else:
                # Use a mix: random, perturbed best, and corner-like points
                r = np.random.rand()
                if r < 0.3:
                    x_start = np.random.uniform(-4.5, 4.5, size=n)
                elif r < 0.6:
                    x_start = best_x + np.random.normal(0, 1.5, size=n)
                    x_start = np.clip(x_start, -4.9, 4.9)
                else:
                    # Corner-like: mix of ±3 values
                    x_start = np.sign(np.random.randn(n)) * np.random.uniform(2, 4, size=n)
                    x_start = np.clip(x_start, -4.9, 4.9)
            
            # Initialize Simplex around starting point with adaptive step
            step = 2.5 if restart < 4 else 1.2
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            for i in range(n):
                simplex[i + 1] = x_start
                simplex[i + 1, i] += step
            
            # Clip to box
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            init_ok = True
            for i in range(n + 1):
                if calls >= budget or time.time() > deadline:
                    init_ok = False
                    break
                x = simplex[i]
                fx = func(x)
                fvals[i] = fx
                calls += 1
            
            if not init_ok or calls == 0:
                break
            
            # Update global best from initial evaluation
            local_best_idx = np.argmin(fvals)
            if fvals[local_best_idx] < best_fx:
                best_fx = fvals[local_best_idx]
                best_x = simplex[local_best_idx].copy()
            
            # Run Nelder-Mead for this restart
            # More iterations per restart, but with periodic polishing
            max_iter = min(remaining // (n + 2), 30)
            polish_interval = 5  # polish every 5 NM iterations
            last_polish_iter = 0
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Periodic coordinate polish during NM iterations
                if it - last_polish_iter >= polish_interval and calls + 15 < budget:
                    x_polished, f_polished = coordinate_polish(simplex[0], fvals[0], budget, max_steps=2)
                    last_polish_iter = it
                    # If polish improved, update simplex best
                    if f_polished < fvals[0]:
                        simplex[0] = x_polished
                        fvals[0] = f_polished
                        if f_polished < best_fx:
                            best_fx = f_polished
                            best_x = x_polished.copy()
                
                # Sort simplex by fitness (best first)
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Update best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()
                
                # Check for convergence: if simplex is very small, polish and break
                spread = np.max(np.abs(simplex - simplex[0]), axis=0)
                if np.all(spread < 1e-3):
                    # Do a final polish before breaking
                    if calls + 15 < budget and time.time() < deadline:
                        x_polished, f_polished = coordinate_polish(simplex[0], fvals[0], budget, max_steps=3)
                        if f_polished < best_fx:
                            best_fx = f_polished
                            best_x = x_polished.copy()
                    break
                
                # Identify worst point
                worst_idx = n
                worst_x = simplex[worst_idx]
                worst_f = fvals[worst_idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # 1. Reflection
                x_ref = centroid + alpha * (centroid - worst_x)
                x_ref = np.clip(x_ref, -5, 5)
                
                if calls >= budget or time.time() > deadline:
                    break
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < fvals[0]:
                    # Reflection is better than best, try Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    
                    if calls >= budget or time.time() > deadline:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        break
                    
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[worst_idx] = x_exp
                        fvals[worst_idx] = f_exp
                    else:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        
                elif f_ref < fvals[n - 1]:
                    # Reflection is better than worst, but worse than second best
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    
                else:
                    # Reflection is worse than worst, try Contraction
                    x_con = centroid + rho * (worst_x - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < worst_f:
                        simplex[worst_idx] = x_con
                        fvals[worst_idx] = f_con
                    else:
                        # Shrinkage: only shrink towards best
                        for i in range(1, n + 1):
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
            
            # Final coordinate polish after Nelder-Mead completes
            if calls < budget and time.time() < deadline:
                x_polished, f_polished = coordinate_polish(simplex[0], fvals[0], budget, max_steps=3)
                if f_polished < best_fx:
                    best_fx = f_polished
                    best_x = x_polished.copy()

        # Final global polish on the overall best
        if calls < budget and time.time() < deadline:
            x_final, f_final = coordinate_polish(best_x, best_fx, budget, max_steps=3)
            if f_final < best_fx:
                best_fx = f_final
                best_x = x_final.copy()
        
        return best_x, best_fx
