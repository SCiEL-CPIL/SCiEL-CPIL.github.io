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
        
        # Step 2: Determine the optimal vertex efficiently.
        # For a linear function, f(x) = c + v^T x.
        # f(0) = c.
        # f(5 e_i) = c + 5 v_i.
        # If f(5 e_i) < f(0), then v_i < 0, so optimal x_i is -5.
        # If f(5 e_i) > f(0), then v_i > 0, so optimal x_i is +5.
        # We only need 1 probe per dimension to determine the sign.
        
        sign = np.zeros(d)  # +1 for +5, -1 for -5
        sign_known = np.zeros(d, dtype=bool)
        
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Probe dimension i at +5 (other dims at 0, as we haven't determined their signs yet
            # and we want to isolate v_i relative to origin)
            # Actually, we can probe at 5 e_i regardless of other signs if we compare to f(0)?
            # No, f(5 e_i) = c + 5 v_i. This is independent of other coords.
            # So we can just use x_probe = 5 e_i.
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            calls_made += 1
            
            # Determine sign based on comparison to f0 (origin)
            # f_probe = c + 5 v_i
            # f0 = c
            # If f_probe < f0, v_i < 0 -> sign -1 (x_i = -5)
            # If f_probe > f0, v_i > 0 -> sign +1 (x_i = +5)
            # If equal, sign doesn't matter, pick +1.
            if f_probe < f0:
                sign[i] = -1.0
            else:
                sign[i] = 1.0
            sign_known[i] = True
            
            # Update best if probe itself is better (it won't be if we pick the opposite sign,
            # but if v_i < 0, f_probe < f0, so it is better. If v_i > 0, f_probe > f0, so not better).
            if f_probe < f_best:
                f_best = f_probe
                x_best = x_probe.copy()
            
            # Evaluate the partial vertex with known signs
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
        # Note: The last partial vertex evaluation in the loop above IS the full vertex.
        # But if we broke out of the loop early, we might not have evaluated it.
        if np.all(sign_known) and calls_made < self.budget and time.time() < deadline:
            x_vertex = 5.0 * sign
            # Check if we already evaluated this exact point?
            # To be safe and simple, we just evaluate it. It's 1 call.
            # However, if the loop completed, the last partial vertex WAS the full vertex.
            # We can skip this if we detect it, but it's safer to just check if we have budget.
            # Given the loop evaluates partials, the last one is the full vertex.
            # So this step is redundant if the loop finished.
            # But if we broke out, we might not have the full vertex.
            # Let's just evaluate it if we haven't already (hard to track easily without storing last point).
            # Actually, we can just check if x_best is not the vertex? No, x_best is just the best found.
            # Let's just evaluate it. It's cheap.
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
