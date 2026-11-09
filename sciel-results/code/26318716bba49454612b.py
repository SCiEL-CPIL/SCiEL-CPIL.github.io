"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead Simplex Optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s/330s kill

        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        def run_nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return
            
            # Initial simplex
            x0 = np.asarray(x0)
            simplex = [x0.copy()]
            
            # Perturbation for other vertices
            # Use a small perturbation in each dimension
            # Standard Nelder-Mead uses 0.05 * x0 or 0.00025 if x0 is 0
            # We use a fixed step size relative to the box size
            step = 0.5 
            
            for i in range(n):
                vertex = x0.copy()
                if abs(vertex[i]) > 0.001:
                    vertex[i] += 0.05 * abs(vertex[i])
                else:
                    vertex[i] += step
                simplex.append(vertex)
            
            # Evaluate initial simplex
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline:
                    return
                f_val = wrapped_func(simplex[i])
                # Update best in case the initial points are better than current best
                # wrapped_func already updates best_f and best_x
            
            # Sort simplex by function value
            # We need to store function values
            f_values = [wrapped_func(simplex[i]) for _ in range(n+1)] # Wait, this double evaluates if not careful.
            # Let's re-implement the loop to be safe and efficient
            
            # Actually, let's just use scipy's minimize with a wrapper that counts calls.
            # But scipy's minimize might not stop exactly on budget.
            # We can pass a callback? No, scipy minimize doesn't have a standard callback for stopping based on external state easily without modifying the function.
            # The wrapped_func raises an exception on time limit. We need a budget limit too.
            
            # Let's implement a manual Nelder-Mead or use a safer wrapper.
            # The previous attempt in the diff used scipy but scored poorly (0.39). 
            # Why? Maybe it got stuck in local minima or didn't restart enough?
            # Or maybe the exception handling broke it?
            
            # Let's try a robust manual implementation or a better scipy usage.
            # SciPy's Nelder-Mead is quite robust. 
            # Key issue: "More than 500 calls ... makes the whole result invalid".
            # So we MUST strictly limit calls.
            
            # Let's restart the logic.
            # 1. Define a wrapper that stops after `max_calls` for this specific run.
            # 2. Use scipy.optimize.minimize with method='Nelder-Mead'.
            # 3. Catch exceptions or check budget.

            pass # Placeholder, logic below

        # Phase 1: Coarse Exploration / Multiple Restarts
        # Strategy:
        # 1. Start at origin.
        # 2. Run Nelder-Mead for a fixed budget (e.g., 50 calls).
        # 3. If no improvement or stuck, restart from a random point or the best point found so far with a perturbation.
        # 4. Repeat until budget is exhausted.

        rng = np.random.default_rng(12345)
        
        # Evaluate origin first
        x0 = np.zeros(n)
        f0 = wrapped_func(x0)
        
        current_best_x = best_x.copy()
        current_best_f = best_f
        
        # Parameters for restarts
        calls_per_run = 40  # Calls per Nelder-Mead run
        max_runs = budget // calls_per_run
        
        for run_idx in range(max_runs):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Define a local wrapper that respects the remaining budget for this run
            # We need to ensure we don't exceed total budget.
            
            # SciPy's minimize doesn't have a strict "max_evals" that stops exactly 
            # without potentially doing one more step or internal checks.
            # However, if we wrap the function to raise an exception when 
            # calls_made + local_calls >= budget, we can catch it.
            
            local_calls = 0
            max_local_calls = min(calls_per_run, budget - calls_made)
            
            def local_wrapped_func(x):
                nonlocal local_calls
                if local_calls >= max_local_calls:
                    raise Exception("Local budget exceeded")
                local_calls += 1
                return wrapped_func(x)
            
            try:
                # Start from the current best point
                res = minimize(
                    local_wrapped_func, 
                    current_best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000, # High, will be stopped by local budget
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                # If it finishes normally (converged), update best
                if res.success or res.fun < current_best_f:
                    # res.x might be slightly outside bounds, clip it
                    res_x = np.clip(res.x, -5, 5)
                    # We already evaluated res.x inside local_wrapped_func? 
                    # Not necessarily the final point if it converged via internal check.
                    # But wrapped_func updates best_f/best_x on every call.
                    # So best_x/best_f are already updated if res.x was better.
                    pass
                    
            except Exception:
                # Budget exceeded or time limit
                pass
            
            # Check if we made progress
            if best_f < current_best_f:
                current_best_f = best_f
                current_best_x = best_x.copy()
            else:
                # No progress, restart from a random point or perturbed best
                # Random restart helps escape local minima in rotated Rosenbrock
                if run_idx > 0:
                    # Mix of random and perturbed best
                    if rng.random() < 0.5:
                        current_best_x = rng.uniform(-5, 5, n)
                    else:
                        current_best_x = best_x + rng.standard_normal(n) * 0.5
                        current_best_x = np.clip(current_best_x, -5, 5)
                    
                    # Evaluate the new start point
                    if calls_made < budget and time.time() < deadline:
                        f_new = wrapped_func(current_best_x)
                        if f_new < best_f:
                            current_best_f = best_f
                            current_best_x = best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
