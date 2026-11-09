import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Phase 1: Multi-start (1+1)-ES with restarts for global exploration
        # Use 60% of budget for global phase
        global_budget = int(0.6 * budget)
        local_budget = budget - global_budget
        
        # Multi-start (1+1)-ES
        n_starts = 5
        budget_per_start = global_budget // n_starts
        
        for s in range(n_starts):
            if time.time() > deadline or calls >= global_budget:
                break
            
            # Initialize start point
            if s == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-4, 4, d)
            
            try:
                f_x = safe_func(x)
            except StopIteration:
                break
            
            sigma = 1.5
            stagnation = 0
            max_stagnation = 30
            
            while calls < global_budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.03
                    stagnation = 0
                else:
                    sigma *= 0.85
                    stagnation += 1
                    if stagnation > max_stagnation:
                        break
                
                if sigma < 1e-6:
                    break
        
        # Phase 2: L-BFGS-B local refinement
        if best_x is not None and calls < budget and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 100, 'maxfun': min(200, budget - calls)}
                )
                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.asarray(result.x, dtype=np.float64).copy()
            except (StopIteration, Exception):
                pass
        
        # Phase 3: Nelder-Mead high-precision polish
        if best_x is not None and calls < budget and time.time() < deadline:
            try:
                nm_result = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(300, budget - calls),
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if nm_result.fun < best_f:
                    best_f = float(nm_result.fun)
                    best_x = np.asarray(nm_result.x, dtype=np.float64).copy()
            except (StopIteration, Exception):
                pass
        
        # Final (1+1)-ES micro-polish if budget remains
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.01
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.02
                else:
                    sigma *= 0.9
                    if sigma < 1e-10:
                        break
        
        return best_x, best_f
