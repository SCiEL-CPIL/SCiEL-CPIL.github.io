"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead based optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # State to track best solution found
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        call_count_lock = [0] # Use a list to allow mutation in nested scope if needed, though not strictly necessary here

        # Wrapper to count calls and handle potential exceptions in func
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                val = func(x)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                calls_made += 1
                return 1e6 # Return large value on error

        # Multi-Start Strategy
        # We need to balance exploration (finding good basins) and exploitation (converging).
        # Nelder-Mead is local. We will use a few random starts.
        # Budget is 500. 
        # Let's reserve some budget for initial random sampling to find good starting points.
        # Then use the rest for local optimization.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Sampling to find promising regions
        # Let's spend ~10-20% of budget on random sampling.
        n_random = max(10, budget // 10)
        
        # Generate random points in [-5, 5]^dim
        # We might want to bias towards center or use a grid? 
        # For Gallagher 101, peaks are scattered. Uniform random is a safe bet.
        random_points = rng.uniform(-5, 5, size=(n_random, dim))
        
        # Evaluate random points
        for x in random_points:
            if calls_made >= budget:
                break
            safe_func(x)
            
        # Phase 2: Local Optimization from best random points
        # Select top K random points as initial guesses for Nelder-Mead.
        # We need to track which points were evaluated in Phase 1 to pick the best ones.
        # Since safe_func updates best_f but doesn't store all values, let's re-evaluate or store them.
        # Actually, we can just store the values from Phase 1.
        
        # Let's restart the logic slightly to track values properly for selection.
        # I will re-implement the loop to store (x, f) pairs for the random phase.
        
        # Reset state for clean implementation
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def counting_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                val = float(func(x))
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                calls_made += 1
                return 1e6

        # Phase 1: Random Sampling
        n_random = max(10, budget // 10)
        random_points = rng.uniform(-5, 5, size=(n_random, dim))
        random_fits = []
        
        for x in random_points:
            if calls_made >= budget:
                break
            f_val = counting_func(x)
            random_fits.append((f_val, x))
            
        # Sort random fits to get best starting points
        random_fits.sort(key=lambda item: item[0])
        
        # Phase 2: Nelder-Mead from top K starts
        # How many starts? 
        # Each NM run can take variable number of calls.
        # Let's pick top 3 starts.
        n_starts = 3
        starts = [x for _, x in random_fits[:n_starts]]
        
        # If we didn't do any random eval (shouldn't happen), fallback
        if not starts:
            starts = [np.zeros(dim)]
            counting_func(starts[0])
            
        # Remaining budget for local optimization
        # We will run NM sequentially from the best start.
        # If budget remains, try next start.
        
        for start_x in starts:
            if calls_made >= budget:
                break
            
            # Define options for Nelder-Mead
            # maxiter is not the same as maxfev. 
            # We want to limit function evaluations.
            # scipy.optimize.minimize with method='Nelder-Mead' uses maxiter.
            # To strictly limit calls, we might need to wrap it or use a large maxiter and hope it converges or stops.
            # However, scipy's NM doesn't have a maxfev parameter directly in the same way as some other optimizers.
            # It stops when maxiter is reached or convergence criteria are met.
            # A standard NM in 5D might take ~50-100 evals to converge.
            # Let's set maxiter to a reasonable number that fits in the remaining budget.
            
            remaining_budget = budget - calls_made
            # Heuristic: maxiter ~ remaining_budget / (dim + 1) ? 
            # Or just set a fixed maxiter like 200 and let it run, relying on budget check in counting_func.
            # But if it doesn't stop, it will hit the budget cap in counting_func and return best_f, effectively stopping progress but wasting iterations?
            # No, counting_func returns best_f, which might cause NM to think it converged or just keep going with bad values.
            # It's better to limit maxiter based on remaining budget.
            
            # Let's say we allow up to 100 iterations for the first start, less for subsequent.
            # Or simply: maxiter = min(100, remaining_budget // 2)
            
            max_iter = min(100, max(1, remaining_budget // 2))
            
            try:
                res = minimize(
                    counting_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                # Update best if result is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
                
        # Phase 3: Use any remaining budget for final random perturbations around best_x
        while calls_made < budget:
            # Small step local search
            step_size = 0.1
            x_cand = best_x + rng.normal(0, step_size, size=dim)
            # Clip to bounds
            x_cand = np.clip(x_cand, -5, 5)
            f_cand = counting_func(x_cand)
            
            # If we improved, we continue. If not, maybe reduce step size?
            # For simplicity, just keep sampling.
            
        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = counting_func(best_x)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
