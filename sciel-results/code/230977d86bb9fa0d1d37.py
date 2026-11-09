"""
Strategy: Multi-Start Adaptive Coordinate Descent (ACD) with Brent's Method.
Rationale: The Discus function (f11) is characterized by a very flat plateau and a narrow, deep valley.
Nelder-Mead can struggle with the anisotropy and the flat regions, often taking many steps to orient itself.
Adaptive Coordinate Descent (ACD) treats the problem as a sequence of 1D optimizations along the coordinate axes.
By using Brent's method for each 1D slice, we can exploit the unimodality along the axes (locally) to find minima efficiently.
The 'Adaptive' part involves maintaining a step size (or interval width) for each dimension, which is expanded if the minimum is at the boundary and contracted if it's inside.
This allows the search to quickly jump across the flat plateau to the basin of attraction and then refine the position within the narrow valley.
Multi-start is used to handle the rotation/shift and potential local minima, but the core engine is ACD.
"""

import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def brent_1d(x_base, axis, a, b, max_iter=10):
            """
            Perform 1D minimization along 'axis' of x_base.
            Uses scipy.optimize.minimize_scalar with method='brent'.
            Returns (new_x, f_val, calls_used).
            """
            nonlocal n_calls
            
            def objective(t):
                if n_calls >= budget:
                    return float('inf')
                x_temp = x_base.copy()
                x_temp[axis] = t
                f_val = func(x_temp)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_temp.copy()
                n_calls += 1
                return f_val
            
            # We need to manage the call count carefully.
            # minimize_scalar doesn't have a strict call limit in older versions, 
            # but we can wrap it or just rely on the global budget check inside objective.
            # However, if budget is hit, objective returns inf, which might confuse Brent.
            # Better to check budget before calling minimize_scalar and break if close.
            
            if n_calls >= budget - 10:
                return x_base, objective(x_base[axis]), 0 # Just evaluate current

            res = minimize_scalar(objective, bounds=(a, b), method='brent', options={'xatol': 1e-6})
            
            # minimize_scalar returns a result object.
            # If it failed or hit limit, res.x might be valid or not.
            # We track n_calls inside objective.
            
            new_x = x_base.copy()
            new_x[axis] = res.x
            
            # Ensure we didn't go out of bounds (Brent should respect bounds, but just in case)
            new_x[axis] = np.clip(new_x[axis], -5.0, 5.0)
            
            # We don't know exactly how many calls Brent used internally without a counter in the wrapper,
            # but objective updates n_calls.
            
            return new_x, res.fun, 0 # calls tracked globally

        def acd_run(x_start, max_calls):
            """
            Run Adaptive Coordinate Descent from x_start.
            Maintains step sizes for each dimension.
            """
            nonlocal n_calls, best_x, best_f
            
            x = x_start.copy()
            # Initialize step sizes (interval half-widths)
            # Start with a reasonably large step to explore the plateau
            steps = np.full(dim, 2.0) 
            # If we are near the boundary, reduce step
            for i in range(dim):
                if x[i] > 3.0:
                    steps[i] = 2.0 - (x[i] - 3.0)
                elif x[i] < -3.0:
                    steps[i] = 2.0 - (-3.0 - x[i])
            
            start_n = n_calls
            
            # Evaluate starting point
            f_curr = call_f(x)
            if f_curr is None:
                return

            while n_calls - start_n < max_calls and time.time() < deadline - 5:
                improved = False
                
                for i in range(dim):
                    if n_calls - start_n >= max_calls:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    # Define bounds for 1D search
                    # We search in [x[i] - steps[i], x[i] + steps[i]]
                    # Clipped to [-5, 5]
                    a = max(-5.0, x[i] - steps[i])
                    b = min(5.0, x[i] + steps[i])
                    
                    # If the interval is too small, skip or shrink step
                    if b - a < 1e-5:
                        # Shrink step to avoid numerical issues
                        steps[i] *= 0.5
                        continue

                    # Perform 1D minimization
                    new_x, f_new, _ = brent_1d(x, i, a, b)
                    
                    # Check if we improved
                    if f_new < f_curr:
                        # Accept the new point
                        x = new_x
                        f_curr = f_new
                        
                        # Check if the minimum is at the boundary
                        # If so, expand the step size for next iteration
                        # If not, shrink it to refine
                        if np.isclose(new_x[i], a, atol=1e-4) or np.isclose(new_x[i], b, atol=1e-4):
                            # Minimum is at boundary, expand search
                            steps[i] *= 1.5
                            # Cap expansion to avoid going too far
                            steps[i] = min(steps[i], 5.0)
                        else:
                            # Minimum is inside, shrink search
                            steps[i] *= 0.7
                            # Floor for step size
                            steps[i] = max(steps[i], 1e-4)
                        
                        improved = True
                    else:
                        # No improvement. Shrink step size to try a finer local search
                        # or keep it if we just expanded.
                        # If we didn't improve, the current step might be too large or we are at a local min.
                        # Shrink to refine.
                        steps[i] *= 0.5
                        steps[i] = max(steps[i], 1e-4)

                # If no dimension improved in a full cycle, we might be stuck.
                # For ACD, this is a strong signal to stop this run.
                if not improved:
                    break

        # Multi-start strategy
        # 1. Start at 0
        # 2. Subsequent starts at random points, but informed by previous bests if possible?
        # For f11, random starts in [-4, 4] are good.
        
        # Budget allocation:
        # First run: 100 calls
        # Subsequent runs: 50 calls each
        
        # Run 1
        acd_run(np.zeros(dim), 100)
        
        # Subsequent runs
        while n_calls < budget and time.time() < deadline - 10:
            # Generate a random start point
            # Bias towards the center slightly? Or uniform?
            # Uniform in [-4, 4] is safe.
            x_start = rng.uniform(-4.0, 4.0, dim)
            
            # Optionally, if we have a good best_x, we could perturb it?
            # But ACD is local. Random restarts are better for global coverage.
            
            # Allocate remaining budget, capped at 50 for this run to allow more restarts
            remaining = budget - n_calls
            max_calls_for_run = min(50, remaining)
            
            if max_calls_for_run < 10:
                break
                
            acd_run(x_start, max_calls_for_run)

        return best_x, best_f
