"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: Multi-start Derivative-Free Coordinate Descent (DFCD) with
multi-resolution grid scanning. This approach systematically explores the
bi-modal landscape by decoupling dimensions and using progressive refinement
from coarse to fine grids, which helps escape local minima traps.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin for 300s limit
        d = self.dim
        budget = self.budget

        # Best solution tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls_made
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Generate diverse starting points using Sobol sequence
        n_starts = 20
        rng = np.random.default_rng(42)
        
        # Sobol for diverse starts
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_samples = sampler.random(n_starts)
            start_points = sobol_samples * 10.0 - 5.0
        except Exception:
            start_points = rng.uniform(-5, 5, size=(n_starts, d))

        # Ensure we try the origin and a few other key points
        start_points[0] = np.zeros(d)
        start_points[1] = np.ones(d) * 2.5
        start_points[2] = np.ones(d) * -2.5

        # Multi-resolution steps for coordinate descent
        # Start coarse, get finer
        steps = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

        def coordinate_descent(x_start, step_sizes):
            """
            Perform coordinate descent from x_start using the provided step sizes.
            Returns the best point found and the number of calls used.
            """
            nonlocal best_x, best_f, calls_made, deadline
            
            x_curr = x_start.copy()
            f_curr = eval_func(x_curr)
            
            for step in step_sizes:
                if calls_made >= budget - 5 or time.time() > deadline - 2.0:
                    break
                
                improved = True
                max_iter = 50  # Limit iterations per step size to prevent infinite loops
                iter_count = 0
                
                while improved and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    
                    if calls_made >= budget - 5 or time.time() > deadline - 2.0:
                        break
                    
                    for i in range(d):
                        if calls_made >= budget - 5 or time.time() > deadline - 2.0:
                            break
                        
                        # Try positive step
                        x_try = x_curr.copy()
                        x_try[i] += step
                        f_try = eval_func(x_try)
                        
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                        
                        # Try negative step
                        x_try = x_curr.copy()
                        x_try[i] -= 2 * step  # Note: x_curr might have changed, so use original logic? 
                        # Actually, standard coordinate descent usually checks +step and -step from the current best in this dim.
                        # Let's stick to a simple pattern: check +step, if better, move. Check -step, if better, move.
                        # The above logic moves to +step if better. Then checks -step from the NEW position? 
                        # That might overshoot. 
                        # Better: Check +step from x_curr. If better, update x_curr. 
                        # Then check -step from x_curr. If better, update x_curr.
                        
                        # Let's re-implement the inner loop slightly more carefully to ensure we don't miss the minimum between steps.
                        # For now, the simple +/- check is a valid DFCD variant.
                        
                        # We already did +step. Now -step.
                        x_try = x_curr.copy()
                        x_try[i] -= step
                        f_try = eval_func(x_try)
                        
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                    
                    # If no improvement in a full pass, break the while loop for this step size
                    if not improved:
                        break
                
                # If no improvement at this step size, we could break early, but sometimes smaller steps are needed.
                # However, if a larger step didn't improve, a smaller step might still help if we are on a ridge.
                # But typically, if a step doesn't improve, we proceed to the next (smaller) step.
            
            return x_curr, f_curr

        # Run DFCD from each starting point
        # We allocate budget roughly evenly, but stop early if budget is exhausted.
        # To be safe, we limit the number of starts if budget is small, but here budget is 500.
        # 20 starts * (1 initial + 11 steps * 5 dims * 2 evals/step * few iters)
        # 11 steps * 5 dims * 2 evals = 110 evals per full pass.
        # If we do 1 pass per step size, that's 110 evals per start.
        # 20 starts * 110 = 2200 evals. Too many.
        
        # We need to be more efficient.
        # Let's reduce the number of steps or starts.
        # Or, we can stop early if no improvement.
        
        # Revised strategy:
        # 1. Coarse scan: Large steps, few iterations.
        # 2. Fine scan: Small steps, more iterations on the best point.
        
        # Let's try:
        # Phase 1: 10 starts, steps [2.0, 1.0, 0.5], max_iter 5.
        # Phase 2: 5 starts (best from phase 1 + new), steps [0.2, 0.1, 0.05, 0.02, 0.01], max_iter 10.
        
        # To keep it simple and robust, let's just run the loop but break if no improvement for a step size.
        # And limit the total calls.
        
        # Let's adjust the steps list to be shorter for the initial runs.
        coarse_steps = [2.0, 1.0, 0.5]
        fine_steps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        # Phase 1: Coarse
        for start_idx in range(min(10, n_starts)):
            if calls_made >= budget - 50 or time.time() > deadline - 10.0:
                break
            
            x0 = start_points[start_idx].copy()
            coordinate_descent(x0, coarse_steps)
        
        # Phase 2: Fine on the best point and a few others
        if best_x is not None:
            # Re-run fine steps on the best point
            if calls_made < budget - 20 and time.time() < deadline - 5.0:
                coordinate_descent(best_x, fine_steps)
            
            # Also try fine steps on the second best start point (if distinct)
            # To keep it simple, just run on best_x again with even finer steps if budget allows
            if calls_made < budget - 10 and time.time() < deadline - 2.0:
                very_fine_steps = [0.002, 0.001, 0.0005]
                coordinate_descent(best_x, very_fine_steps)

        return best_x, best_f
