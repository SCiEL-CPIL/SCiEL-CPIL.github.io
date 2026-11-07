"""
Black-box continuous optimizer for BBOB f16 (Weierstrass).
Two-phase approach: (1+1)-ES global search + Nelder-Mead local refinement.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        
        n = self.dim
        total_budget = self.budget
        
        # Reserve budget for Nelder-Mead polish
        nm_budget = min(100, total_budget // 5)
        es_budget = total_budget - nm_budget
        
        # Phase 1: (1+1)-ES global search
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Adaptive step size with 1/5 success rule
        sigma = 1.5
        success_window = 10
        successes = 0
        window_count = 0
        
        while calls < es_budget and time.time() < deadline:
            # Generate candidate
            step = rng.randn(n) * sigma
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
            else:
                successes -= 0  # no change, but we count the attempt
            
            window_count += 1
            if window_count >= success_window:
                # 1/5 success rule
                if successes / window_count > 0.2:
                    sigma *= 1.5
                elif successes / window_count < 0.2:
                    sigma *= 0.75
                successes = 0
                window_count = 0
            
            # If sigma becomes too small, try a larger random jump
            if sigma < 1e-4:
                sigma = 0.5
                x = best_x + rng.randn(n) * 0.1
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
        
        # Phase 2: Nelder-Mead local refinement
        if calls < total_budget and time.time() < deadline:
            nm_calls = total_budget - calls
            x_nm = best_x.copy()
            fx_nm = best_f
            
            # Initialize simplex
            simplex = [x_nm.copy()]
            fvals = [fx_nm]
            
            # Create simplex vertices
            for i in range(n):
                v = x_nm.copy()
                v[i] += 0.5 if v[i] < 4.5 else -0.5
                v = np.clip(v, -5, 5)
                fv = func(v)
                simplex.append(v)
                fvals.append(fv)
                calls += 1
                if fv < best_f:
                    best_x = v.copy()
                    best_f = fv
            
            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            
            while calls < total_budget and time.time() < deadline:
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if len(set([f'{f:.15e}' for f in fvals])) <= 2:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_x = xr.copy()
                    best_f = fr
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_x = xe.copy()
                        best_f = fe
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_x = xc.copy()
                        best_f = fc
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_f:
                                best_x = simplex[i].copy()
                                best_f = fi
        
        return best_x, best_f
