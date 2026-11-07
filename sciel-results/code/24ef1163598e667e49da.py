import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid multi-start optimizer: ES + L-BFGS-B for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, lower, upper)
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def es_phase(x_start, fx_start, max_calls, sigma_init=1.5):
            """Run a short (1+1)-ES phase with 1/5 success rule."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x_start, fx_start
            
            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            window_size = 8
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
                        sigma *= 1.25
                    elif rate < 0.2:
                        sigma /= 1.25
                
                # Update current position
                if success:
                    x = y
                    fx = fy
                
                # Prevent sigma explosion or vanishing
                sigma = np.clip(sigma, 1e-6, 8.0)
            
            return x, fx

        def lbfgsb_phase(x_start, max_calls, ftol=1e-10, gtol=1e-7):
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
                        'maxiter': 25, 
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

        # Phase 1: Diverse multi-start exploration
        # Generate diverse starting points
        starts = []
        
        # Start 0: Origin
        starts.append(np.zeros(dim))
        
        # Start 1: Small random
        starts.append(rng.uniform(-2, 2, dim))
        
        # Start 2: Larger random
        starts.append(rng.uniform(-4, 4, dim))
        
        # Start 3: Full range random
        starts.append(rng.uniform(-5, 5, dim))
        
        # Start 4-7: More random points
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all starts
        start_results = []
        for x0 in starts:
            if calls >= budget or time.time() > deadline:
                break
            fx0 = wrapped_func(x0)
            start_results.append((fx0, x0.copy()))
        
        if not start_results:
            return best_x, best_f
        
        # Sort by fitness
        start_results.sort(key=lambda x: x[0])
        
        # Phase 2: For top starts, do ES + L-BFGS-B refinement
        # Use top 4 starts with ES exploration followed by L-BFGS-B
        n_refine = min(4, len(start_results))
        per_start_budget = max(15, (budget - calls - 30) // n_refine)
        
        for i in range(n_refine):
            if calls >= budget or time.time() > deadline:
                break
            
            fx0, x0 = start_results[i]
            
            # Short ES phase to explore local neighborhood
            es_budget = min(20, per_start_budget // 2)
            if es_budget >= 5:
                x_es, fx_es = es_phase(x0, fx0, es_budget, sigma_init=1.0)
            else:
                x_es, fx_es = x0, fx0
            
            # L-BFGS-B refinement from ES result
            remaining_for_this = min(per_start_budget - es_budget, budget - calls)
            if remaining_for_this >= 10:
                x_refined = lbfgsb_phase(x_es, remaining_for_this, ftol=1e-10, gtol=1e-7)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 3: Intensive refinement of the best point found
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Multiple refinement attempts with decreasing tolerances
                for attempt in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    tol = [1e-12, 1e-14, 1e-15][min(attempt, 2)]
                    maxfev_this = max(5, remaining // (3 - attempt))
                    
                    try:
                        res = scipy.optimize.minimize(
                            wrapped_func, 
                            best_x, 
                            method='L-BFGS-B', 
                            bounds=bounds,
                            options={
                                'maxiter': 30, 
                                'maxfev': maxfev_this, 
                                'ftol': tol, 
                                'gtol': 1e-9, 
                                'disp': False
                            }
                        )
                    except Exception:
                        pass
        
        # Phase 4: Final polish with tightest tolerances
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 50, 
                            'maxfev': remaining, 
                            'ftol': 1e-15, 
                            'gtol': 1e-10, 
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
