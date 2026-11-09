import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lower, upper = -5.0, 5.0
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), lower, upper)
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
            calls_used += 1
            if np.isnan(f_val) or np.isinf(f_val):
                return np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Multi-start with (1+1)-ES for global exploration
        # Weierstrass is highly multimodal, so we use multiple restarts
        # with limited budget each, then pick the best basin for refinement
        
        n_restarts = 5
        es_budget_per_restart = max(30, budget // (n_restarts + 2))
        
        # First, do a quick random sampling to find promising regions
        n_samples = min(20, budget // 10)
        for _ in range(n_samples):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            safe_call(x)
        
        # Now do multi-start ES from the best points found
        # Keep top few starting points
        es_starts = []
        # We don't have a list of all points, so we'll do a few more targeted samples
        # and use the current best as one start, plus random ones
        
        for restart in range(n_restarts):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Choose starting point: mix of current best and random
            if restart == 0:
                x = best_x.copy()
            elif restart == 1:
                # Perturb best point
                x = best_x + rng.normal(0, 1.0, dim)
                x = np.clip(x, lower, upper)
            else:
                x = rng.uniform(-4.0, 4.0, dim)
            
            fx = safe_call(x)
            if np.isinf(fx):
                continue
            
            # (1+1)-ES with adaptive sigma
            sigma = 1.0
            sigma_min = 0.005
            sigma_max = 3.0
            success_count = 0
            total_attempts = 0
            target_success_rate = 0.2
            
            for i in range(es_budget_per_restart):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, lower, upper)
                
                fy = safe_call(y)
                if np.isinf(fy):
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                else:
                    pass
                
                total_attempts += 1
                
                if total_attempts >= 8:
                    success_rate = success_count / total_attempts
                    if success_rate > target_success_rate:
                        sigma *= 1.15
                    elif success_rate < target_success_rate:
                        sigma *= 0.85
                    success_count = 0
                    total_attempts = 0
                
                sigma = np.clip(sigma, sigma_min, sigma_max)
        
        # Phase 2: Local refinement with Nelder-Mead from the best point found
        remaining_budget = budget - calls_used
        if remaining_budget > 5 and not np.isinf(best_f):
            x_start = best_x.copy()
            
            # Create initial simplex
            simplex_step = 0.05
            simplex = [x_start.copy()]
            for i in range(dim):
                p = x_start.copy()
                p[i] += simplex_step
                p = np.clip(p, lower, upper)
                simplex.append(p)
            
            nm_calls = 0
            
            def wrapped_func(x):
                nonlocal nm_calls, calls_used, best_x, best_f
                if calls_used >= budget or time.time() > deadline:
                    return 1e10
                x = np.clip(np.asarray(x, dtype=np.float64), lower, upper)
                try:
                    f_val = func(x)
                except Exception:
                    f_val = 1e10
                calls_used += 1
                nm_calls += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return 1e10
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            
            def callback(xk):
                if calls_used >= budget or time.time() > deadline:
                    return True
                return False
            
            try:
                result = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': simplex
                    },
                    callback=callback
                )
            except Exception:
                pass
        
        # Phase 3: If we still have budget, do a few more NM restarts from perturbed best
        remaining_budget = budget - calls_used
        if remaining_budget > 10 and not np.isinf(best_f):
            for attempt in range(3):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Perturb the best point slightly
                x_start = best_x + rng.normal(0, 0.1, dim)
                x_start = np.clip(x_start, lower, upper)
                
                simplex_step = 0.02
                simplex = [x_start.copy()]
                for i in range(dim):
                    p = x_start.copy()
                    p[i] += simplex_step
                    p = np.clip(p, lower, upper)
                    simplex.append(p)
                
                def wrapped_func2(x):
                    nonlocal calls_used, best_x, best_f
                    if calls_used >= budget or time.time() > deadline:
                        return 1e10
                    x = np.clip(np.asarray(x, dtype=np.float64), lower, upper)
                    try:
                        f_val = func(x)
                    except Exception:
                        f_val = 1e10
                    calls_used += 1
                    if np.isnan(f_val) or np.isinf(f_val):
                        return 1e10
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                    return f_val
                
                def callback2(xk):
                    if calls_used >= budget or time.time() > deadline:
                        return True
                    return False
                
                try:
                    result = scipy.optimize.minimize(
                        wrapped_func2,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining_budget // 4, 50),
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True,
                            'initial_simplex': simplex
                        },
                        callback=callback2
                    )
                except Exception:
                    pass
        
        return best_x, best_f
