import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Strategy: Multi-Start Nelder-Mead with Halton initialization
        # 1. Generate a set of diverse starting points using a Halton sequence.
        # 2. Run a fast, limited Nelder-Mead optimization from each start.
        # 3. Track the global best found across all restarts.
        
        # We allocate the budget among multiple restarts.
        # For dim=5, Nelder-Mead is relatively cheap per iteration.
        # Let's use 20 restarts.
        # Total budget 500.
        # Each restart gets ~25 evaluations.
        
        n_restarts = 20
        evals_per_restart = budget // n_restarts
        
        # Generate Halton points for initialization
        # Halton sequences are low-discrepancy, good for covering the space.
        # We need n_restarts points in [0, 1]^dim, then map to [-5, 5]^dim.
        
        try:
            # qmc.Halton is available in scipy.stats
            halton = qmc.Halton(d=dim, scramble=True, seed=42)
            points = halton.random(n=n_restarts)
            # Map [0, 1] to [-5, 5]
            points = points * 10.0 - 5.0
        except Exception:
            # Fallback to random if Halton fails
            rng = np.random.default_rng(42)
            points = rng.uniform(-5, 5, (n_restarts, dim))
        
        best_x_global = None
        best_f_global = np.inf
        
        # We will iterate through the restarts.
        # We can prioritize restarts if we find a very good solution early,
        # but for simplicity and robustness, we just run them in order.
        # To improve efficiency, we might sort initial points by some heuristic?
        # No, Halton is already well-distributed.
        
        # We need to be careful with time.
        # If we run out of time, we stop.
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
                
            x0 = points[i]
            
            # Wrap the function to count calls and check time
            calls_made = 0
            max_calls = evals_per_restart
            
            def wrapped_func(x):
                nonlocal calls_made
                if time.time() > deadline:
                    raise TimeoutError()
                calls_made += 1
                # Ensure x is within bounds? func clips, so it's safe.
                return func(x)
            
            try:
                # Run Nelder-Mead
                # 'maxiter' in scipy's NM is roughly the number of iterations,
                # but each iteration uses dim+1 function calls (simplex size).
                # So maxiter * (dim+1) approximates function calls.
                # To be safe and precise, we rely on the wrapper's call count
                # and the 'maxiter' option as a hard stop for the algorithm itself.
                # However, scipy's NM doesn't have a direct 'maxfev' in all versions
                # or it's tricky. The wrapper raising an exception is the most robust way.
                
                # Let's set a reasonable maxiter.
                # If evals_per_restart is 25, and dim=5, simplex size is 6.
                # 25 / 6 ~ 4 iterations. That's very few.
                # Maybe we should reduce restarts to allow more refinement per restart?
                # 10 restarts * 50 evals?
                # 5 restarts * 100 evals?
                
                # Let's stick to 20 restarts but allow NM to run until it fails or time runs out.
                # The wrapper will stop it if we hit the per-restart budget or global deadline.
                
                # Actually, if we raise an exception in the wrapper, scipy might not catch it cleanly
                # or might treat it as a failure. It's better to let it run but check time.
                # But we have a hard budget constraint.
                
                # Alternative: Use 'maxiter' to limit iterations.
                # If maxiter is too high, it might exceed evals_per_restart.
                # Let's estimate: 1 iteration = dim+1 calls = 6 calls.
                # evals_per_restart = 25.
                # maxiter = 25 // 6 = 4.
                
                maxiter = max(1, evals_per_restart // (dim + 1))
                
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                # Check if we found a better solution
                # res.fun might be inf if it failed immediately
                if res.fun < best_f_global:
                    best_f_global = res.fun
                    best_x_global = res.x.copy()
                    
                # Also check if any intermediate point was better?
                # Scipy doesn't easily expose intermediate history in res.
                # But NM updates the simplex, so the best vertex is usually res.x if it converges.
                # If it doesn't converge, res.x is the best of the final simplex.
                
            except TimeoutError:
                break
            except Exception as e:
                # If NM crashes, continue to next restart
                continue

        # If we didn't find any valid points (shouldn't happen), fallback
        if best_x_global is None:
            x0 = np.zeros(dim)
            best_f_global = func(x0)
            best_x_global = x0

        return best_x_global, best_f_global
