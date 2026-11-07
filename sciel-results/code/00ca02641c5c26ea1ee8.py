import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid ES + L-BFGS-B optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def es_phase(x_start, fx_start, max_calls, sigma_init=2.0):
            """Run a short (1+1)-ES phase with 1/5 success rule."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x_start, fx_start
            
            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            window_size = 10
            successes = []
            
            for _ in range(max_calls):
                if calls >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy >= np.inf:
                    break
                
                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # Update success history for 1/5 rule
                success = fy < fx
                successes.append(1 if success else 0)
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma based on success rate (1/5 rule)
                if len(successes) == window_size:
                    rate = sum(successes) / window_size
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.2:
                        sigma /= 1.3
                
                # Update current position
                if success:
                    x = y
                    fx = fy
                
                # Prevent sigma explosion or vanishing
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            return x, fx

        def lbfgsb_phase(x_start, max_calls, ftol=1e-12, gtol=1e-8):
            """Run L-BFGS-B from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x_start
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'maxfev': max_calls, 
                        'ftol': ftol, 
                        'gtol': gtol, 
                        'disp': False
                    }
                )
                if res.x is not None and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                return res.x if res.x is not None else x_start
            except Exception:
                return x_start

        # Phase 1: Initial ES exploration to find a good basin
        x0 = rng.uniform(-5, 5, dim)
        fx0 = wrapped_func(x0)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Run ES for ~100 calls to explore
        es_budget = min(100, budget - calls)
        x_es, fx_es = es_phase(x0, fx0, es_budget)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: L-BFGS-B refinement from the best ES point
        remaining = budget - calls
        # Allocate most remaining budget to L-BFGS-B
        lbfgsb_budget = max(20, int(remaining * 0.7))
        x_refined = lbfgsb_phase(x_es, lbfgsb_budget)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 3: Check if we should do another ES burst + L-BFGS-B
        # If we've made progress, do a short ES from the current best
        remaining = budget - calls
        if remaining > 30:
            # Short ES burst from the best point found so far
            es_budget2 = min(30, remaining // 3)
            x_es2, fx_es2 = es_phase(best_x, best_f, es_budget2, sigma_init=1.0)
            
            if calls >= budget or time.time() > deadline:
                return best_x, best_f
            
            # L-BFGS-B from the new ES result
            remaining2 = budget - calls
            if remaining2 > 10:
                x_refined2 = lbfgsb_phase(x_es2, remaining2, ftol=1e-14, gtol=1e-10)
        
        return best_x, best_f
