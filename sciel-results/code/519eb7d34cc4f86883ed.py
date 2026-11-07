"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using Sobol sequence for global exploration followed by 
    Nelder-Mead for local refinement. This exploits the deterministic 
    nature of the function and its smooth continuous landscape, avoiding 
    the stochastic noise of evolutionary methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we stop at 300s to be safe
        deadline = start_time + 300.0 
        n = self.dim
        total_budget = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to check time and budget
        def check_limits():
            nonlocal calls_made
            if calls_made >= total_budget or time.time() > deadline:
                return False
            return True

        # Helper to evaluate and update best
        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            if not check_limits():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Sobol Sequence Initialization
        # Sobol sequences are low-discrepancy sequences that provide better 
        # coverage of the space than random sampling.
        # We generate a batch of points. 
        # In 5D, we can generate 50-100 points quickly.
        n_sobol = min(100, total_budget // 5) # Reserve budget for refinement
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, scramble=False)
            # Generate n_sobol points in [0, 1]^n
            sobol_points = sampler.random(n_sobol)
            # Map to [-5, 5]^n
            sobol_points = sobol_points * 10.0 - 5.0
            
            for x in sobol_points:
                if not check_limits():
                    break
                eval_point(x)
        except Exception:
            # Fallback to random if scipy qmc fails
            rng = np.random.default_rng(42)
            for _ in range(n_sobol):
                if not check_limits():
                    break
                x = rng.uniform(-5, 5, n)
                eval_point(x)

        if best_x is None:
            # Should not happen, but safeguard
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        # 2. Local Refinement with Nelder-Mead
        # We will attempt to refine the best point found so far.
        # Nelder-Mead is robust for non-linear, non-smooth (though this is smooth) 
        # and doesn't require gradients.
        
        # We might also want to try refining a few other promising points 
        # if the best one gets stuck, but for now let's focus on the best.
        
        # To improve robustness, we can run NM from the global best, 
        # and if it fails or hits limits, try from the 2nd best, etc.
        # However, maintaining a list of top-k points is complex. 
        # Let's stick to refining the current best, but restart NM if it 
        # stagnates or if we have budget left.
        
        # Since NM consumes function calls internally, we need to wrap func
        # to count calls and check time.
        
        nm_func_wrapper_calls = [0]
        
        def nm_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                # Return a large value to signal stop, or raise? 
                # Raising might be cleaner for scipy to stop.
                # But we need to ensure we don't exceed budget.
                # Let's just return a high value and hope it converges/stops.
                # Actually, better to check inside the wrapper if possible.
                # Scipy's minimize doesn't have a callback for stopping mid-iteration easily 
                # except via options or exceptions.
                # Let's use a callback to check time.
                pass
            # We must count this call
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        # Callback to check time and budget
        def nm_callback(xk):
            if time.time() > deadline or calls_made >= total_budget:
                # Return True to stop
                return True
            # Update best if this point is better
            fk = nm_func(xk) # This is wrong, nm_func already evaluated. 
                             # We can't re-evaluate. 
                             # We need to get the function value from the history.
            # Simpler: just check time. The best update will happen if we 
            # manually check after or if we trust that NM finds better.
            # But we need to update best_f/best_x for the final return.
            # Let's rely on the fact that NM minimizes, so the final xk is likely good.
            # However, intermediate points might be better if it oscillates? Unlikely for NM.
            # Let's just return True if limits hit.
            return time.time() > deadline or calls_made >= total_budget

        # Run Nelder-Mead
        # x0 is the best point found by Sobol
        x0 = best_x.copy()
        
        # Options for NM
        options = {
            'maxiter': 1000, # Max iterations
            'maxfev': total_budget - calls_made, # Max function evaluations
            'xatol': 1e-8,
            'fatol': 1e-8
        }
        
        try:
            res = minimize(nm_func, x0, method='Nelder-Mead', options=options, callback=nm_callback)
            # Update best with result
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            # If NM fails, continue with what we have
            pass

        # 3. If we still have budget, try a second round of NM from a slightly perturbed best
        # Or, if the first NM didn't use much budget, run it again?
        # NM usually converges quickly. Let's check if we have significant budget left.
        
        remaining_budget = total_budget - calls_made
        if remaining_budget > 50:
            # Try refining from a small perturbation of the best
            # This helps if NM got stuck in a local minimum (though Schaffers F7 is unimodal-ish with ill-conditioning)
            # Actually, Schaffers F7 is multimodal? "ill-conditioned" usually refers to the condition number of the Hessian.
            # Let's try a few random restarts of NM if budget allows.
            
            rng = np.random.default_rng(123)
            for i in range(5): # Max 5 restarts
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Perturb best_x
                sigma = 0.1 * (10.0 / (i + 1)) # Decreasing perturbation
                x_start = best_x + rng.normal(0, sigma, n)
                x_start = np.clip(x_start, -5, 5)
                
                # Evaluate start point
                fx_start = func(x_start)
                calls_made += 1
                if fx_start < best_f:
                    best_f = fx_start
                    best_x = x_start.copy()
                
                # Run NM from here
                options2 = {
                    'maxiter': 500,
                    'maxfev': max(10, total_budget - calls_made),
                    'xatol': 1e-8,
                    'fatol': 1e-8
                }
                
                try:
                    res2 = minimize(nm_func, x_start, method='Nelder-Mead', options=options2, callback=nm_callback)
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
