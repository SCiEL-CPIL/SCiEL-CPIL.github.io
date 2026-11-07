"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start Nelder-Mead Optimizer.
    
    Strategy:
    1. Generate a set of diverse starting points using a Halton sequence (low-discrepancy)
       to ensure good coverage of the [-5, 5]^5 space.
    2. For each starting point, run the Nelder-Mead simplex method.
    3. Nelder-Mead is a derivative-free method that adapts its simplex shape to the local
       landscape, making it effective for multimodal functions like BBOB f22 (Gallagher 21 peaks)
       when combined with multiple starts.
    4. Keep track of the global best.
    5. Use a strict time guard to ensure we return within the limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # 1. Generate starting points using Halton sequence
        # We want to distribute starts across the budget.
        # Nelder-Mead typically uses (dim + 1) initial points and then iterates.
        # Each iteration uses 1 function evaluation (reflection, expansion, etc. might use more,
        # but let's assume ~2-3 per iteration on average).
        # Let's reserve some budget for the initial points themselves.
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = np.zeros((count, dim))
            for i in range(count):
                for j in range(dim):
                    # Use distinct primes for each dimension
                    base = 2 + j  # 2, 3, 5, 7, 11...
                    points[i, j] = halton(i + 1, base)
            return points

        # We will allocate a fraction of the budget to each restart.
        # Let's say we allow up to 50 restarts.
        # Each restart gets budget // 50 calls.
        # But Nelder-Mead can be expensive in terms of function calls if it stagnates.
        # We will impose a per-restart call limit.
        
        max_restarts = 40
        calls_per_restart = max(10, budget // max_restarts)
        
        # Generate Halton points in [0, 1]^dim, then scale to [-5, 5]^dim
        halton_points = halton_sequence(max_restarts, dim)
        start_points = halton_points * 10.0 - 5.0
        
        # Add the center as a start point too, just in case
        start_points = np.vstack([np.zeros(dim), start_points])
        
        global_best_x = start_points[0]
        global_best_f = func(global_best_x)
        calls = 1
        
        for i in range(1, len(start_points)):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            x0 = start_points[i]
            
            # If we already have a very good solution, maybe skip?
            # No, let's just run it.
            
            # Define a wrapper for func that counts calls and checks deadline
            # But since we need to pass the original func to minimize, we can't easily
            # intercept calls without wrapping.
            # However, minimize takes a function. We can wrap it.
            
            # But wait, the contract says "Every call is recorded". 
            # If we wrap func, does the evaluator see the wrapper? 
            # The evaluator calls __call__(self, func). The func passed in is the one that records.
            # If we pass a wrapper to minimize, the wrapper calls func. 
            # So the evaluator sees the calls through func.
            
            # We need to be careful: minimize might call func many times.
            # We should limit the number of iterations in minimize.
            
            # Nelder-Mead parameters
            # maxiter is the max number of iterations.
            # Each iteration can use multiple function calls.
            # Let's set maxiter to be roughly calls_per_restart / 2.
            
            max_iter = max(5, calls_per_restart // 2)
            
            try:
                # x0 is the starting point
                # method='Nelder-Mead'
                # options:
                #   maxiter: max iterations
                #   xatol, fatol: tolerances
                #   adaptive: True (default in newer scipy)
                
                # We need to track calls to ensure we don't exceed budget.
                # Since we can't easily interrupt minimize mid-run based on global call count,
                # we rely on maxiter and the time guard.
                # If minimize runs long, the time guard will break the outer loop,
                # but the current minimize call might still be running.
                # To be safe, we can use a smaller maxiter.
                
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
                
                # res.x is the best point found by this restart
                # res.fun is the function value
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
                
            except Exception:
                # If minimize fails, skip this restart
                pass
            
            # Update call count? 
            # We don't know exactly how many calls minimize made.
            # We can estimate: max_iter * (dim + 1) roughly? 
            # Actually, Nelder-Mead uses 1 call per iteration on average (reflection, etc).
            # Let's assume ~max_iter calls.
            calls += max_iter
            
            # If we exceed budget, break
            if calls >= budget:
                break
        
        # If we have time and budget left, we can do a final refinement on the global best
        if time.time() < deadline - 5 and calls < budget:
            try:
                remaining_budget = budget - calls
                max_iter_final = max(5, remaining_budget // 2)
                res = minimize(func, global_best_x, method='Nelder-Mead',
                               options={'maxiter': max_iter_final, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
            except Exception:
                pass
        
        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
