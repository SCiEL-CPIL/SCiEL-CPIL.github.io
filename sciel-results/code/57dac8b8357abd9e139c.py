import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise StopIteration
            try:
                f_val = func(np.array(x, dtype=np.float64))
                if not np.isfinite(f_val):
                    f_val = 1e10
            except Exception:
                f_val = 1e10
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Initial exploration to find a good basin
        # Lunacek has a global min and a local min.
        # We do a quick random scan to avoid starting in a terrible spot.
        n_scan = min(20, budget)
        for _ in range(n_scan):
            x = rng.uniform(-5, 5, n)
            try:
                safe_func(x)
            except StopIteration:
                break
            if time.time() > deadline:
                break

        if eval_count >= budget:
            return best_x, best_f

        # Phase 2: (1+1)-ES with 1/5 success rule and restarts
        # This is a robust strategy for BBOB functions.
        current_x = best_x.copy()
        current_f = best_f
        sigma = 2.0  # Start with a larger step to cover the domain
        
        # Parameters for 1/5 rule
        c_sigma = 1.0
        target_success = 0.2
        window = 10
        successes = 0
        total = 0
        
        # Restart logic
        no_improve_count = 0
        max_no_improve = 30
        restart_count = 0
        max_restarts = 5

        while eval_count < budget and time.time() < deadline:
            z = rng.standard_normal(n)
            candidate = np.clip(current_x + sigma * z, -5, 5)
            
            try:
                cand_f = safe_func(candidate)
            except StopIteration:
                break
            
            # Update success statistics
            total += 1
            if cand_f < current_f:
                successes += 1
                current_x = candidate
                current_f = cand_f
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adjust sigma based on 1/5 rule
            if total >= window:
                success_rate = successes / total
                if success_rate > target_success:
                    sigma *= (1 + c_sigma * (success_rate - target_success))
                else:
                    sigma *= (1 + c_sigma * (success_rate - target_success))
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-4, 5.0)
                
                successes = 0
                total = 0

            # Restart if stuck or sigma too small
            if no_improve_count > max_no_improve or sigma < 0.01:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Restart from best point found so far with a larger sigma
                current_x = best_x.copy()
                current_f = best_f
                sigma = 2.0
                no_improve_count = 0
                successes = 0
                total = 0
                
                # Optionally, jump to a random point if stuck too long
                if restart_count > 2:
                    current_x = np.clip(best_x + rng.normal(0, 2.0, n), -5, 5)
                    try:
                        current_f = safe_func(current_x)
                    except StopIteration:
                        break

        return best_x, best_f
