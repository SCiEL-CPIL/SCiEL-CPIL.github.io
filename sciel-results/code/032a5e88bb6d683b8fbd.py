import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin for 300s limit
        
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
        
        # Phase 1: Smart initialization - find the valley center
        # Bent Cigar has a narrow valley; we need to locate it
        num_initial = 15
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
        
        # Phase 2: Powell's Method - excellent for narrow valleys
        def powell_func(x):
            return safe_func(x)
            
        try:
            res = minimize(
                powell_func, 
                x0, 
                method='Powell',
                bounds=[(-5, 5)] * dim,
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-14,
                    'disp': False
                }
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
            
        # Phase 3: Adaptive refinement if budget remains
        if calls < budget and time.time() < deadline - 10:
            # Use Nelder-Mead with adaptive restart for final polish
            x_nm = best_x.copy()
            f_nm = best_f
            
            # Initialize simplex with adaptive delta based on current best
            # Start with moderate delta, shrink if no improvement
            delta = 0.5
            max_nm_iterations = 30
            
            for attempt in range(3):  # Up to 3 NM restarts with shrinking delta
                if time.time() > deadline or calls >= budget:
                    break
                
                # Initialize simplex around current best
                simplex = [x_nm.copy()]
                simplex_f = [f_nm]
                
                for i in range(dim):
                    x_i = x_nm.copy()
                    x_i[i] += delta
                    x_i = np.clip(x_i, -5.0, 5.0)
                    f_i = safe_func(x_i)
                    simplex.append(x_i)
                    simplex_f.append(f_i)
                    
                    if time.time() > deadline or calls >= budget:
                        break
                
                if len(simplex) < dim + 1:
                    break
                
                # Nelder-Mead loop
                rho = 1.0
                chi = 2.0
                gamma = 0.5
                sigma = 0.5
                improvement = 0.0
                
                for _ in range(max_nm_iterations):
                    if time.time() > deadline or calls >= budget:
                        break
                    
                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]
                    
                    if simplex_f[-1] - simplex_f[0] < 1e-12:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    xr = centroid + rho * (centroid - simplex[-1])
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = safe_func(xr)
                    
                    if fr < simplex_f[0]:
                        xe = centroid + chi * (xr - centroid)
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = safe_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                            improvement += (simplex_f[0] - fe) if fe < simplex_f[0] else 0
                            if fe < best_f:
                                best_f = fe
                                best_x = xe.copy()
                                x_nm = xe.copy()
                                f_nm = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                            if fr < best_f:
                                best_f = fr
                                best_x = xr.copy()
                                x_nm = xr.copy()
                                f_nm = fr
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                            x_nm = xr.copy()
                            f_nm = fr
                    else:
                        xc = centroid + gamma * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_func(xc)
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                            if fc < best_f:
                                best_f = fc
                                best_x = xc.copy()
                                x_nm = xc.copy()
                                f_nm = fc
                        else:
                            # Shrink simplex
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = safe_func(simplex[i])
                            
                            indices = np.argsort(simplex_f)
                            simplex = [simplex[i] for i in indices]
                            simplex_f = [simplex_f[i] for i in indices]
                            
                            if simplex_f[0] < best_f:
                                best_f = simplex_f[0]
                                best_x = simplex[0].copy()
                                x_nm = simplex[0].copy()
                                f_nm = simplex_f[0]
                
                # If no significant improvement, shrink delta for next attempt
                if improvement < 1e-10:
                    delta *= 0.3
                    if delta < 1e-6:
                        break
        
        return best_x, best_f
