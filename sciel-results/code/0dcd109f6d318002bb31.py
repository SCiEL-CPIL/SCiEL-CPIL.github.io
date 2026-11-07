import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_powell(x0, maxiter, ftol=1e-8):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Powell',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': maxiter,
                        'ftol': ftol
                    }
                )
            except Exception:
                pass

        # Phase 1: Dense Sobol quasi-random exploration
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        num_sobol = 100
        sobol_points = sobol.random_base2(m=7)  # 128 points, take first 100
        sobol_points = sobol_points[:num_sobol]
        # Map [0,1]^n to [-5,5]^n
        sobol_points = sobol_points * 10.0 - 5.0
        
        for x in sobol_points:
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(x)
        
        # Phase 2: Local optimization from top candidates
        # Find the best few points from Sobol exploration and polish them
        # We'll use the best point found so far as primary start
        if best_x is not None and calls < budget and time.time() < deadline:
            # Powell's method for smooth convergence
            run_powell(best_x, maxiter=50, ftol=1e-8)
            
            # Nelder-Mead for final polish
            if calls < budget and time.time() < deadline:
                run_nm(best_x, max_iter=30, xatol=1e-9, fatol=1e-11)
        
        # Phase 3: Additional local searches from promising regions
        if best_x is not None and calls < budget and time.time() < deadline:
            # Try a few perturbations around the best point
            for scale in [0.5, 0.2, 0.1]:
                if calls >= budget or time.time() >= deadline:
                    break
                x0 = best_x + rng.normal(0, scale, size=n)
                x0 = np.clip(x0, -5.0, 5.0)
                run_powell(x0, maxiter=30, ftol=1e-9)
                if calls < budget and time.time() < deadline:
                    run_nm(x0, max_iter=20, xatol=1e-10, fatol=1e-12)
        
        # Phase 4: Final high-precision polish
        if best_x is not None and calls < budget and time.time() < deadline:
            run_nm(best_x, max_iter=50, xatol=1e-11, fatol=1e-13)

        return best_x, best_f
