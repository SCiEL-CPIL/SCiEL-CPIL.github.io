import numpy as np
import time
from scipy.optimize import minimize

class BudgetExhausted(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # Wrapper to count calls and enforce budget
        def counting_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                raise BudgetExhausted()
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Numerical gradient using central differences
        # Cost: 2*d calls
        def counting_grad(x):
            nonlocal calls
            if calls + 2 * d > budget or not time_ok():
                raise BudgetExhausted()
                
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box to handle boundaries, though Discus is smooth inside
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = counting_func(x_plus)
                f_minus = counting_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Generate candidate starting points
        # 1. Center
        # 2-10. Random points
        rng = np.random.RandomState(12345) # Fixed seed for reproducibility of strategy, but points are relative
        candidates = [np.zeros(d)]
        for _ in range(9):
            candidates.append(rng.uniform(-4.0, 4.0, d))
        
        # Phase 1: Evaluate all candidates to find the best start
        # This costs 10 calls.
        # We do this to ensure the first L-BFGS-B run starts from the best point.
        for pt in candidates:
            if calls >= budget or not time_ok():
                break
            try:
                f_val = counting_func(pt)
            except BudgetExhausted:
                break
        
        # Sort candidates by their function values (using best_f logic is tricky as we don't store per-candidate)
        # Actually, we can just re-evaluate? No, we can't re-evaluate without spending calls.
        # We should store the values.
        
        # Let's restart the logic slightly to store candidate values.
        # We already spent 10 calls.
        # We need to know which candidate was best.
        # Let's re-implement Phase 1 to store results.
        
        # Reset state for a cleaner implementation
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        def counting_func_v2(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                raise BudgetExhausted()
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        def counting_grad_v2(x):
            nonlocal calls
            if calls + 2 * d > budget or not time_ok():
                raise BudgetExhausted()
                
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = counting_func_v2(x_plus)
                f_minus = counting_func_v2(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Generate candidates
        rng = np.random.RandomState(12345)
        candidates = [np.zeros(d)]
        for _ in range(9):
            candidates.append(rng.uniform(-4.0, 4.0, d))
        
        # Evaluate candidates
        cand_vals = []
        for pt in candidates:
            if calls >= budget or not time_ok():
                break
            try:
                f_val = counting_func_v2(pt)
                cand_vals.append((f_val, pt))
            except BudgetExhausted:
                break
        
        # Sort by value
        cand_vals.sort(key=lambda x: x[0])
        
        # Phase 2: Run L-BFGS-B from best candidates
        bounds = [(-5, 5)] * d
        
        for f_val, start_pt in cand_vals:
            if calls >= budget or not time_ok():
                break
            
            try:
                res = minimize(
                    counting_func_v2, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=counting_grad_v2, 
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'maxls': 20
                    }
                )
            except BudgetExhausted:
                break
            except Exception:
                # If minimize fails for other reasons, continue to next candidate
                continue
                
        return best_x, best_f
