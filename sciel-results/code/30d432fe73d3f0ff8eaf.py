import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random Global Sampling
        # Use 40% of budget for global exploration
        n_global = int(budget * 0.4)
        global_results = []
        
        # Use a seeded RNG for reproducibility if needed, but random is fine
        rng = np.random.default_rng(42)
        
        for i in range(n_global):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            x_try = rng.uniform(-5.0, 5.0, size=n)
            f_try = try_eval(x_try)
            if f_try is not None:
                global_results.append((f_try, x_try.copy()))

        if not global_results:
            return best_x, best_f

        # Sort by fitness
        global_results.sort(key=lambda x: x[0])
        
        # Phase 2: Select Top-K Candidates for Local Refinement
        # Ensure diversity: pick top candidates that are not too close to each other
        candidates = []
        min_dist = 1.0 # Minimum distance between restart points
        
        for f_val, x_val in global_results:
            if len(candidates) >= 5:
                break
            # Check distance to existing candidates
            is_diverse = True
            for _, x_cand in candidates:
                if np.linalg.norm(x_val - x_cand) < min_dist:
                    is_diverse = False
                    break
            if is_diverse:
                candidates.append((f_val, x_val))
        
        # If we didn't find 5 diverse points, just take the top 5 overall
        if len(candidates) < 5:
            candidates = [(f, x) for f, x in global_results[:5]]
            
        # Phase 3: Local Refinement using Nelder-Mead
        # Allocate remaining budget to each candidate
        remaining_budget = budget - total_calls
        if remaining_budget <= 0:
            return best_x, best_f
            
        n_local = len(candidates)
        budget_per_candidate = max(10, int(remaining_budget / n_local))
        
        for f_init, x_init in candidates:
            if total_calls >= budget - 10 or time.time() > deadline - 10:
                break
            
            # Define local function for scipy
            def local_func(x_local):
                nonlocal total_calls
                if total_calls >= budget:
                    return 1e10
                if time.time() > deadline - 5:
                    return 1e10
                x_clipped = np.clip(x_local, -5.0, 5.0)
                f_val = float(func(x_clipped))
                total_calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val

            # Run Nelder-Mead
            # options: maxiter controls the number of iterations
            # We want to use up to budget_per_candidate evaluations
            # NM uses 1 eval per iteration roughly (actually n+1 for init, then 1-2 per step)
            # So maxiter should be around budget_per_candidate / 2
            max_iter = max(10, budget_per_candidate // 2)
            
            try:
                res = minimize(local_func, x_init, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                # The best point might have been found during the optimization
                # res.x is the final point, but we track best_x in closure
            except Exception:
                pass

        return best_x, best_f
