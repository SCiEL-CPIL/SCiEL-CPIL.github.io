import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        calls_made = 0
        
        def check_time():
            return time.time() > deadline
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or check_time():
                return None
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_made += 1
            return val
        
        # Phase 1: Global Exploration with Coordinate Descent refinement
        # Use fewer initial points but refine each one with coordinate descent
        # to exploit ridge structure early on
        n_explore = 15
        points = rng.uniform(-5, 5, (n_explore, n))
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_explore):
            if check_time() or calls_made >= budget:
                break
            
            x = points[i].copy()
            f_val = safe_call(x)
            if f_val is None:
                break
            
            # Quick coordinate descent refinement
            for _cd_iter in range(3):
                if check_time() or calls_made >= budget:
                    break
                improved = False
                for d in range(n):
                    if check_time() or calls_made >= budget:
                        break
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        step = 0.5 * sign
                        x_trial[d] += step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            break
                        if f_trial < f_val:
                            x = x_trial
                            f_val = f_trial
                            improved = True
                            # Try to continue in same direction
                            for _inner in range(3):
                                if check_time() or calls_made >= budget:
                                    break
                                x_trial2 = x.copy()
                                x_trial2[d] += step
                                x_trial2 = np.clip(x_trial2, -5, 5)
                                f_trial2 = safe_call(x_trial2)
                                if f_trial2 is None:
                                    break
                                if f_trial2 < f_val:
                                    x = x_trial2
                                    f_val = f_trial2
                                else:
                                    break
                    if check_time() or calls_made >= budget:
                        break
                if not improved:
                    break
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
        
        if best_x is None:
            # Fallback
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            if f0 is None:
                return np.zeros(n), np.inf
            return x0, f0
        
        # Phase 2: (1+1)-ES Refinement from best point
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Initial step size - smaller for sharp ridge
        sigma = 0.5
        
        # Adaptive step size with success rate tracking
        success_count = 0
        total_count = 0
        
        while calls_made < budget:
            if check_time():
                break
            
            # Generate candidate
            z = rng.standard_normal(n)
            x_new = x_curr + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_call(x_new)
            if f_new is None:
                break
            
            if f_new < f_curr:
                # Success
                x_curr = x_new
                f_curr = f_new
                success_count += 1
                total_count += 1
                # Increase step size if success rate is high
                if total_count > 10 and success_count / total_count > 0.5:
                    sigma *= 1.1
                else:
                    sigma *= 1.02
            else:
                # Failure
                total_count += 1
                # Decrease step size
                sigma *= 0.95
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            # Reset success tracking periodically
            if total_count > 50:
                success_count = 0
                total_count = 0
        
        return x_curr, f_curr
