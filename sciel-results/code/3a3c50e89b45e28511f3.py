import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Deterministic space-filling exploration using Halton sequence
        # Use a smaller set of points to leave more budget for local optimization
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r

        # Use 80 Halton points for global exploration
        num_halton = 80
        halton_results = []
        
        for i in range(1, num_halton + 1):
            if calls_made >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            x = np.array([halton(i, base) for base in range(2, dim + 2)])
            x = x * 10.0 - 5.0
            
            fx = eval_func(x)
            halton_results.append((fx, x))

        if not halton_results:
            x = np.zeros(dim)
            eval_func(x)
            return best_x, best_f

        # Sort to find the best point from Halton sampling
        halton_results.sort(key=lambda t: t[0])
        best_halton_f, best_halton_x = halton_results[0]

        # Phase 2: Local optimization using CMA-ES-like strategy with restarts
        # Use multiple restarts from different starting points to handle multimodality
        
        if time.time() < deadline and calls_made < budget - 20:
            rng = np.random.default_rng(12345)
            
            # Try local optimization from the best Halton point and a few other good points
            # Use top 3 Halton points as starting points for local search
            num_starts = min(3, len(halton_results))
            remaining_budget = budget - calls_made
            budget_per_start = max(10, remaining_budget // (num_starts + 1))
            
            for start_idx in range(num_starts):
                if calls_made >= budget - 5 or time.time() > deadline:
                    break
                    
                x_curr = halton_results[start_idx][1].copy()
                f_curr = halton_results[start_idx][0]
                sigma = 0.5  # Start with smaller step size for local refinement
                
                # Run (1+1)-ES with adaptive step size
                success_count = 0
                failure_count = 0
                local_budget = budget_per_start
                
                for step in range(local_budget):
                    if calls_made >= budget - 5 or time.time() > deadline:
                        break
                    
                    y = x_curr + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = eval_func(y)
                    
                    if fy < f_curr:
                        x_curr = y
                        f_curr = fy
                        success_count += 1
                        # Increase step size on success
                        sigma *= 1.1
                    else:
                        failure_count += 1
                        # Decrease step size on failure
                        sigma *= 0.9
                    
                    # Reset adaptation counters periodically
                    if success_count + failure_count >= 20:
                        if success_count > failure_count * 2:
                            sigma *= 1.5
                        elif failure_count > success_count * 2:
                            sigma *= 0.5
                        success_count = 0
                        failure_count = 0

        return best_x, best_f
