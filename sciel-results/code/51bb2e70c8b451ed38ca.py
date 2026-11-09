"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    """Generate the index-th Halton number in the given base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    """Deterministic Halton global search followed by Nelder-Mead local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin before 330s kill

        # Split budget: 70% global, 30% local
        global_budget = int(budget * 0.7)
        local_budget = budget - global_budget

        # 1. Global Search using Halton sequence
        # We want to cover [-5, 5]^dim.
        # Halton sequences generate points in [0, 1]^dim.
        # Map to [-5, 5].
        
        # Primes for first 'dim' dimensions
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        best_x = np.zeros(dim)
        best_fx = np.inf
        calls = 0

        # Start with center
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_fx = fx0

        # Generate Halton points
        # We need 'global_budget' points.
        # Skip index 0 as it is (0,0,...,0) which maps to (-5,-5,...,-5) or similar depending on mapping.
        # Standard Halton starts at 1.
        
        for k in range(1, global_budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            x = np.zeros(dim)
            for d in range(dim):
                # Map [0, 1] to [-5, 5]
                x[d] = -5.0 + 10.0 * _halton(k, primes[d])
            
            fx = func(x)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # 2. Local Polish using Nelder-Mead
        # Use scipy.optimize.minimize if available, otherwise implement simple NM.
        # The prompt says SciPy is available.
        
        if time.time() < deadline and calls < budget:
            # We have local_budget calls remaining, but minimize doesn't allow exact call counting easily
            # without a wrapper. We will wrap func to count calls.
            
            remaining_calls = budget - calls
            nm_calls = [0]
            
            def wrapped_func(x):
                if nm_calls[0] >= remaining_calls:
                    return best_fx # Return current best to stop improvement logic? 
                                   # Actually, better to just raise or return large?
                                   # scipy minimize checks convergence. 
                                   # If we exceed budget, we should stop.
                                   # A clean way is to catch the exception or check time.
                                   # Let's just let it run and break the loop if time/calls exceed.
                                   # But minimize is a single call. 
                                   # We can use a custom callback to check.
                                   
                nm_calls[0] += 1
                # Clip to box to ensure validity
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            # We need to stop minimize if we run out of budget or time.
            # scipy minimize doesn't have a "max_calls" directly, but we can use a callback.
            
            def callback(xk):
                # Check if we are out of time or calls
                if time.time() > deadline or nm_calls[0] >= remaining_calls:
                    raise ValueError("Budget exceeded")
            
            try:
                # Nelder-Mead parameters
                # initial_simplex_size is tricky. 
                # We can let scipy choose default or specify.
                # Default NM in scipy uses a simplex with width 0.05 * x + 0.25 * (1-x)
                # Since our domain is [-5,5], 0.25 might be too small for initial steps if we are far from optimum?
                # But we started from the best global point, so we should be in the basin.
                # Let's try to be a bit more aggressive with initial simplex if possible, 
                # but default is usually okay for local refinement.
                
                # To ensure we don't waste calls on very tiny steps, 
                # and to respect the budget, we rely on the callback.
                
                res = scipy_optimize_minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                              options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': remaining_calls},
                                              callback=callback)
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except ValueError:
                pass # Budget exceeded, continue with best found so far
            except Exception:
                pass # Any other error, keep current best

        return best_x, best_fx

# EVOLVE-BLOCK-END
