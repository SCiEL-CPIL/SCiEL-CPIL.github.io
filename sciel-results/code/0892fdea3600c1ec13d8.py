"""Combined program: (1+1)-ES with Restarts + Nelder-Mead Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Global phase: (1+1)-ES with adaptive step size and restarts from best.
    2. Local phase: Nelder-Mead refinement on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.shape != (n,):
                x_arr = x_arr.reshape(-1)
                if x_arr.size != n:
                    raise ValueError("Wrong dimension")
            f_val = float(func(x_arr))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Allocate 55% of budget to global exploration
        global_budget = int(self.budget * 0.55)
        
        # Initial point - use origin as a reasonable start for Schwefel-like
        x = np.zeros(n)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 2.0
        stagnation = 0
        max_stagnation = 40
        
        # Track best in current run for restarts
        run_best_x = x.copy()
        run_best_f = fx
        
        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if sigma > 4.0:
                    sigma = 4.0
                stagnation = 0
                if fx < run_best_f:
                    run_best_f = fx
                    run_best_x = x.copy()
            else:
                sigma *= 0.92
                if sigma < 0.05:
                    sigma = 0.05
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-4, 4, n)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.5
                    stagnation = 0
                    # Reduce threshold for more aggressive restarts later
                    max_stagnation = max(25, int(max_stagnation * 0.7))

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local optimization
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            # Nelder-Mead uses ~dim+1 evaluations per iteration roughly
            # Be conservative with maxiter to stay within budget
            max_iter = max(10, remaining_calls // (n + 3))
            
            try:
                # Nelder-Mead is derivative-free and good for final polish
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass # Keep best from Phase 1 if this fails
        
        # Phase 3: Fine polish with tighter Nelder-Mead if budget remains
        if calls < self.budget - 5 and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 10:
                max_iter = max(5, remaining_calls // (n + 3))
                try:
                    result2 = scipy.optimize.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
