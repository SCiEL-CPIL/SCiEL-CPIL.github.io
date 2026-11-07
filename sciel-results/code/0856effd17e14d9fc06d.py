import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return 1e10
            try:
                calls += 1
                val = func(np.asarray(x, dtype=float))
                return float(val)
            except Exception:
                return 1e10
        
        # Phase 1: Deterministic grid search for guaranteed early progress
        # Evaluate vertices of the hypercube and center points
        grid_points = []
        
        # All 2^5 = 32 vertices of [-5, 5]^5
        for mask in range(2**n):
            pt = np.zeros(n)
            for j in range(n):
                if mask & (1 << j):
                    pt[j] = 5.0
                else:
                    pt[j] = -5.0
            grid_points.append(pt)
        
        # Add center and axis-aligned points
        grid_points.append(np.zeros(n))
        for j in range(n):
            pt_pos = np.zeros(n)
            pt_pos[j] = 5.0
            grid_points.append(pt_pos)
            pt_neg = np.zeros(n)
            pt_neg[j] = -5.0
            grid_points.append(pt_neg)
        
        # Evaluate grid points
        for pt in grid_points:
            if calls >= budget or time.time() > deadline - 10:
                break
            f_val = safe_call(pt)
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        # Phase 2: Multi-start Nelder-Mead from top grid points
        if best_x is not None and calls < budget and time.time() < deadline - 5:
            # Sort grid points by their function values (we need to re-evaluate or track)
            # For simplicity, just use the best point and a few random perturbations
            rng = np.random.default_rng(42)
            
            num_restarts = 8
            calls_per_restart = 30
            
            for i in range(num_restarts):
                if calls >= budget or time.time() > deadline - 10:
                    break
                
                if i == 0:
                    x0 = best_x.copy()
                else:
                    # Perturb the best point
                    x0 = best_x + rng.normal(0, 1.0, n)
                    x0 = np.clip(x0, -5, 5)
                
                # Run Nelder-Mead
                try:
                    res = scipy.optimize.minimize(
                        safe_call,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': calls_per_restart,
                            'xatol': 1e-4,
                            'fatol': 1e-4,
                            'adaptive': True
                        }
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
        
        # Phase 3: Final polish on the best candidate
        if best_x is not None and calls < budget and time.time() < deadline - 5:
            remaining_budget = budget - calls
            max_iter_final = min(remaining_budget, 200)
            
            try:
                res = scipy.optimize.minimize(
                    safe_call,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_final,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Return interface must match current program: return best_x, best_f
        return best_x, best_f
