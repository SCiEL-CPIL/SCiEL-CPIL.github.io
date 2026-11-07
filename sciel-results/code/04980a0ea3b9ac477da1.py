"""Derivative-free Trust-Region BFGS optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DF-TBFGS approach:
    - Uses finite-difference gradients to build a BFGS Hessian approximation.
    - Efficient for smooth functions like f14.
    - Multi-start strategy to avoid local minima.
    - Deadline guard to ensure we return before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 330s kill
        rng = np.random.default_rng(123)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Initial sampling
        # Start at origin and a few random points to identify a good region
        initial_starts = [np.zeros(d)]
        for _ in range(5):
            initial_starts.append(rng.uniform(-4, 4, d))

        for s in initial_starts:
            fx = try_eval(s)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 5:
            return best_x, best_f

        # DF-TBFGS Routine
        def df_t_bfgs(start_x, start_f, max_iters, tol=1e-12):
            nonlocal best_x, best_f, calls_used
            
            x = start_x.copy()
            f = start_f
            H = np.eye(d)  # Inverse Hessian approximation
            step_size = 1.0
            
            # History for BFGS update
            s_prev = None
            g_prev = None
            
            # Initial gradient estimation
            if calls_used + 2*d >= max_calls or time.time() > deadline - 2:
                return x, f
            
            g = np.zeros(d)
            h = 1e-4
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                fx_plus = try_eval(x_plus)
                fx_minus = try_eval(x_minus)
                
                if fx_plus is None or fx_minus is None:
                    return x, f
                
                g[i] = (fx_plus - fx_minus) / (2 * h)
            
            if calls_used >= max_calls or time.time() > deadline - 2:
                return x, f

            for _ in range(max_iters):
                if calls_used >= max_calls or time.time() > deadline - 2:
                    break
                
                # Check convergence
                if np.linalg.norm(g) < tol:
                    break
                
                # Compute step
                step = -H @ g
                step_norm = np.linalg.norm(step)
                
                # Limit step size to avoid large jumps initially
                if step_norm > 1.0:
                    step = step / step_norm
                
                x_new = x + step
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = try_eval(x_new)
                if fx_new is None:
                    break
                
                # Estimate new gradient
                if calls_used + 2*d >= max_calls or time.time() > deadline - 2:
                    break
                
                g_new = np.zeros(d)
                for i in range(d):
                    x_plus = x_new.copy()
                    x_minus = x_new.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    
                    fx_plus = try_eval(x_plus)
                    fx_minus = try_eval(x_minus)
                    
                    if fx_plus is None or fx_minus is None:
                        break
                    
                    g_new[i] = (fx_plus - fx_minus) / (2 * h)
                
                if fx_plus is None or fx_minus is None:
                    break

                # BFGS Update
                s = x_new - x
                y = g_new - g
                sy = s @ y
                
                if sy > 1e-12:
                    # Standard BFGS update
                    Hy = H @ y
                    H = H - (H @ y @ y.T @ H) / (y @ Hy) + (s @ s.T) / sy
                    H = (H + H.T) / 2  # Symmetrize
                    s_prev = s
                    g_prev = y
                
                # Update state
                x = x_new
                f = fx_new
                g = g_new
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            return x, f

        # Main Loop: Multi-start DF-TBFGS
        # We allocate budget for multiple attempts
        # Each attempt uses ~2d calls for gradient + 1 for function + 2d for next gradient
        # So ~4d+1 calls per iteration. With d=5, that's 21 calls.
        # 500 calls allows for ~20 iterations per start.
        
        # Strategy: 
        # 1. Run DF-TBFGS from current best.
        # 2. If budget remains, restart from random points or best with perturbation.
        
        current_best_x = best_x.copy()
        current_best_f = best_f
        
        # Run DF-TBFGS from the best point found in Phase 1
        # We give it a generous iteration count, but it will stop if budget runs out
        x_opt, f_opt = df_t_bfgs(current_best_x, current_best_f, max_iters=100)
        
        # If we have budget left, try a few random restarts
        # This helps if the first start got stuck in a local minimum
        while calls_used < max_calls - 50 and time.time() < deadline - 10:
            # Pick a random start point, possibly near the current best to leverage progress
            # Or purely random to explore
            if rng.random() < 0.5:
                # Local perturbation
                start_x = current_best_x + 1.0 * rng.standard_normal(d)
            else:
                # Global exploration
                start_x = rng.uniform(-4, 4, d)
            
            start_f = try_eval(start_x)
            if start_f is None:
                break
            
            # Run DF-TBFGS from this start
            x_opt, f_opt = df_t_bfgs(start_x, start_f, max_iters=50)
            
            # If we found something better, update current best
            if f_opt < current_best_f:
                current_best_x = x_opt.copy()
                current_best_f = f_opt
        
        return best_x, best_f
# EVOLVE-BLOCK-END
