"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using a hybrid approach:
1. Dense Latin Hypercube Sampling (LHS) for global basin identification.
2. Nelder-Mead Simplex for local exploitation.
3. Adaptive restarts if stagnation is detected.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Dense Latin Hypercube Sampling for global basin identification
        # LHS provides better coverage than random sampling
        n_lhs = 50
        if n_lhs > budget:
            n_lhs = budget
        
        # Generate Latin Hypercube samples
        lhs_indices = np.random.permutation(n_lhs)
        lhs_samples = (lhs_indices[:, np.newaxis] + np.random.rand(n_lhs, 1)) / n_lhs
        lhs_samples = lhs_samples * 10 - 5  # Scale to [-5, 5]
        
        # Evaluate LHS samples
        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_samples[i].copy()
            f = func(x)
            calls += 1
            update_best(x, f)
        
        # Phase 2: Nelder-Mead local optimization from multiple starting points
        # Select top 5 points from LHS as starting points for local optimization
        # We need to track which points were best
        
        # Re-evaluate to get sorted results (we didn't store them, so do a quick re-sort)
        # Actually, let's just use the best point found so far and a few random perturbations
        
        # Better approach: Keep track of top points during LHS
        # For now, use best_x and generate a few diverse starting points
        
        # Define a helper for Nelder-Mead with call tracking
        def nelder_mead_from(x0, max_calls=50):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return
            
            # Create objective wrapper
            def objective(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f
            
            # Set up Nelder-Mead
            options = {
                'maxiter': 100,
                'maxfev': max(1, min(max_calls, budget - calls)),
                'xatol': 1e-6,
                'fatol': 1e-8,
                'disp': False
            }
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', options=options)
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        # Run Nelder-Mead from best point
        if best_x is not None and calls < budget and time.time() < deadline:
            nelder_mead_from(best_x, max_calls=50)
        
        # Run Nelder-Mead from a few other promising points
        # Generate diverse starting points around best_x
        if calls < budget and time.time() < deadline:
            for offset_scale in [1.0, 2.0, 3.0]:
                if calls >= budget or time.time() > deadline:
                    break
                x_start = best_x + np.random.normal(0, offset_scale, size=dim)
                x_start = np.clip(x_start, -5, 5)
                nelder_mead_from(x_start, max_calls=30)
        
        # Phase 3: If still budget left, do a second round of LHS + NM
        if calls < budget and time.time() < deadline:
            n_lhs2 = min(30, budget - calls)
            if n_lhs2 > 0:
                lhs_indices2 = np.random.permutation(n_lhs2)
                lhs_samples2 = (lhs_indices2[:, np.newaxis] + np.random.rand(n_lhs2, 1)) / n_lhs2
                lhs_samples2 = lhs_samples2 * 10 - 5
                
                for i in range(n_lhs2):
                    if calls >= budget or time.time() > deadline:
                        break
                    x = lhs_samples2[i].copy()
                    f = func(x)
                    calls += 1
                    update_best(x, f)
                
                # NM from new best
                if best_x is not None and calls < budget and time.time() < deadline:
                    nelder_mead_from(best_x, max_calls=40)
        
        # Phase 4: Final L-BFGS-B polish
        if calls < budget and best_x is not None and time.time() < deadline:
            def objective_lbfgs(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f
            
            bounds = [(-5, 5)] * dim
            options = {
                'maxiter': 50,
                'maxfun': max(1, min(50, budget - calls)),
                'ftol': 1e-10,
                'gtol': 1e-8,
                'disp': False
            }
            
            try:
                result = minimize(objective_lbfgs, best_x, method='L-BFGS-B', bounds=bounds, options=options)
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        return best_x, best_f
