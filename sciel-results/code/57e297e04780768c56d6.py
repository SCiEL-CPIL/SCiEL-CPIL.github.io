"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid (1+1)-ES + Nelder-Mead for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Phase 1: (1+1)-ES for global exploration
        # Start at origin, large step size to cover [-4, 4]^5
        x = np.zeros(dim)
        sigma = 3.0  # Large initial step
        f_x = func(x)
        calls_made += 1
        update_best(x, f_x)
        
        # Success rule: 1/5 rule
        # If success rate is around 0.2, sigma is well-tuned
        # For (1+1)-ES, we can use a simple adaptation:
        # If improvement, increase sigma slightly; else decrease.
        # But a more robust approach is to track successes over a window.
        # Let's use a simple adaptive step size.
        
        # We will run (1+1)-ES for a portion of the budget, then switch to NM.
        # How much budget for ES? Maybe 60%?
        # Let's say 300 calls for ES, 200 for NM.
        es_budget = 300
        
        # Adaptive (1+1)-ES
        # We'll use a simple rule: if the trial is better, move there and increase sigma by 1.1
        # If not, decrease sigma by 0.9. This is a crude but effective heuristic.
        # A more principled approach: track the success rate over the last N steps.
        
        # Let's use the 1/5 success rule properly.
        # We need a window of steps to estimate the success rate.
        # Let's use a window of 10.
        success_count = 0
        window_size = 10
        success_history = []
        
        while calls_made < es_budget and not check_time():
            # Generate trial
            z = np.random.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < f_x:
                # Success
                x = x_trial
                f_x = f_trial
                update_best(x, f_x)
                success_history.append(1)
            else:
                # Failure
                success_history.append(0)
                
            # Keep history window
            if len(success_history) > window_size:
                success_history.pop(0)
                
            # Adapt sigma based on success rate
            if len(success_history) == window_size:
                success_rate = np.mean(success_history)
                # 1/5 rule: if success rate > 0.2, increase sigma; if < 0.2, decrease
                if success_rate > 0.2:
                    sigma *= 1.05  # Gentle increase
                elif success_rate < 0.2:
                    sigma *= 0.95  # Gentle decrease
                    
                # Clip sigma
                sigma = np.clip(sigma, 0.01, 10.0)
        
        # Phase 2: Nelder-Mead local refinement
        # Use the best point found by ES as the starting point for NM.
        # NM requires dim+1 initial points.
        # We have budget - calls_made calls left.
        
        if best_x is None:
            return best_x, best_f
        
        # Create initial simplex for NM
        # Start with best_x, and perturb each dimension slightly.
        # Perturbation size: maybe 0.1 or current sigma?
        # Let's use a small perturbation, like 0.1.
        perturbation = 0.1
        
        simplex = [best_x.copy()]
        for i in range(dim):
            point = best_x.copy()
            point[i] += perturbation
            point = np.clip(point, -5, 5)
            simplex.append(point)
        
        # Evaluate initial simplex
        # We already have f(best_x) = best_f.
        # We need to evaluate the other dim points.
        # Total calls needed: dim (for the other simplex points).
        
        # Check if we have enough budget
        if calls_made + dim > budget:
            # Not enough budget for a full simplex.
            # We can try a reduced simplex or just stop.
            # For simplicity, if we can't afford dim more calls, we stop.
            # Or we can try to evaluate as many as we can.
            # NM requires a full simplex to start.
            # Let's just stop if we can't afford the initial simplex.
            pass
        else:
            # Evaluate the remaining simplex points
            f_simplex = [best_f]
            for i in range(1, dim + 1):
                if calls_made >= budget or check_time():
                    break
                f_i = func(simplex[i])
                calls_made += 1
                f_simplex.append(f_i)
                update_best(simplex[i], f_i)
            
            # If we couldn't evaluate all points, NM won't work properly.
            # Check if we have dim+1 points.
            if len(f_simplex) == dim + 1 and not check_time() and calls_made < budget:
                # Run Nelder-Mead
                # We'll implement a simple NM loop manually to control the budget.
                # Or use scipy.optimize.minimize with a wrapper.
                # Using scipy is safer and faster.
                
                # Wrapper to count calls
                nm_calls = [0]
                def nm_func(x):
                    nonlocal calls_made
                    if calls_made >= budget or check_time():
                        return float('inf')
                    f_val = func(x)
                    calls_made += 1
                    nm_calls[0] += 1
                    update_best(x, f_val)
                    return f_val
                
                # Options for NM
                # maxiter: limit the number of iterations
                # We have budget - calls_made calls left.
                # Each NM iteration uses 1 call (for the reflection, etc., it might use more).
                # Actually, NM can use multiple calls per iteration.
                # Let's set maxiter to a reasonable number, like 100.
                # And rely on the call budget check inside nm_func.
                
                try:
                    result = minimize(
                        nm_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                except Exception:
                    pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
