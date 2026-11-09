# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    For a linear function f(x) = c + v^T x, the minimum over [-5,5]^d
    is at a vertex. We determine the optimal vertex by finding the sign
    of each gradient component using d+1 evaluations (origin + d axis probes),
    then evaluate that vertex.
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i, evaluate at 5*e_i to determine gradient component v_i.
       After each probe, update the best-known partial vertex and evaluate it
       to make early progress visible to the AOCC metric.
    3. Once all gradient components are known, construct and evaluate the full
       optimal vertex x*.
    4. Stall with the best point found for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Step 2: For each dimension, evaluate both +5 and -5 to determine
        # the optimal sign directly (no finite-difference estimation).
        # Use previously determined optimal coordinates for other dims.
        sign = np.zeros(d)  # +1 for +5, -1 for -5
        sign_known = np.zeros(d, dtype=bool)
        
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Build base point: known dims at their optimal value, unknown at 0
            x_base = np.zeros(d)
            for j in range(d):
                if sign_known[j]:
                    x_base[j] = 5.0 * sign[j]
            
            # Evaluate +5 for dim i
            x_plus = x_base.copy()
            x_plus[i] = 5.0
            f_plus = func(x_plus)
            calls_made += 1
            if f_plus < f_best:
                f_best = f_plus
                x_best = x_plus.copy()
            
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Evaluate -5 for dim i
            x_minus = x_base.copy()
            x_minus[i] = -5.0
            f_minus = func(x_minus)
            calls_made += 1
            if f_minus < f_best:
                f_best = f_minus
                x_best = x_minus.copy()
            
            # Determine optimal sign for dim i
            if f_plus < f_minus:
                sign[i] = 1.0
            else:
                sign[i] = -1.0
            sign_known[i] = True
            
            # Evaluate the updated partial vertex (all known dims at optimal)
            if calls_made < self.budget and time.time() < deadline:
                x_partial = np.zeros(d)
                for j in range(d):
                    if sign_known[j]:
                        x_partial[j] = 5.0 * sign[j]
                fp = func(x_partial)
                calls_made += 1
                if fp < f_best:
                    f_best = fp
                    x_best = x_partial.copy()
        
        # Step 3: If all signs known, evaluate the full optimal vertex
        if np.all(sign_known) and calls_made < self.budget and time.time() < deadline:
            x_vertex = 5.0 * sign
            fv = func(x_vertex)
            calls_made += 1
            if fv < f_best:
                f_best = fv
                x_best = x_vertex.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
