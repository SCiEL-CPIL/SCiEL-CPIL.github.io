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

        def run_lbfgsb(x0, max_iter, gtol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'gtol': gtol
                    }
                )
            except Exception:
                pass

        # Phase 1: Diverse Exploration
        num_exploration = 8
        calls_per_exploration = max(10, (budget - 100) // num_exploration)
        
        starts = [np.zeros(n)]
        for _ in range(num_exploration - 1):
            starts.append(rng.uniform(-4.5, 4.5, size=n))
            
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            # Try L-BFGS-B first (faster for smooth functions), then NM
            run_lbfgsb(x0, max_iter=calls_per_exploration)
            if calls < budget and time.time() < deadline:
                run_nm(x0, max_iter=max(5, calls_per_exploration // 2))

        # Phase 2: Polish
        if best_x is not None and calls < budget and time.time() < deadline:
            # Run L-BFGS-B with tighter tolerance for high-precision polish
            remaining = budget - calls
            if remaining > 10:
                run_lbfgsb(best_x, max_iter=remaining, gtol=1e-14)
            
            # Final NM polish with very tight tolerance
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 5:
                    run_nm(best_x, max_iter=remaining, xatol=1e-12, fatol=1e-14)

        return best_x, best_f
