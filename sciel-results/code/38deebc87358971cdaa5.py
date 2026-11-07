"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Optimizer using Differential Evolution for global search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        # Define bounds for the box [-5, 5]^dim
        bounds = [(-5.0, 5.0) for _ in range(dim)]

        # We will use Differential Evolution which is a global optimizer.
        # We need to manage the evaluation count carefully.
        # DE calls func internally. We wrap it to count and track best.
        
        # State to be updated by the wrapper
        state = {
            'count': 0,
            'best_x': np.zeros(dim),
            'best_f': float('inf'),
            'deadline': deadline
        }

        # Wrapper function to count evaluations and enforce deadline
        def wrapped_func(x):
            if time.time() > state['deadline']:
                # Return a very bad value to stop optimization effectively?
                # Or just return the current best? 
                # Scipy DE might not handle exceptions well. 
                # Returning a large penalty is safer.
                return 1e10
            state['count'] += 1
            val = func(x)
            if val < state['best_f']:
                state['best_f'] = val
                state['best_x'] = x.copy()
            return val

        # Strategy: Run DE with a budget that fits.
        # Popsize = 10, Maxiter = 40 -> ~400 evals.
        # This leaves 100 evals for a potential local refinement or second run.
        
        # Initial run
        try:
            result = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=40, 
                popsize=10, 
                seed=42,
                tol=1e-6,
                init='latinhypercube'
            )
        except Exception as e:
            # If DE fails, fall back to a simple random search or Nelder-Mead
            pass

        # Check time and remaining budget
        remaining_budget = budget - state['count']
        
        if remaining_budget > 50 and time.time() < state['deadline']:
            # Use remaining budget for a local refinement around the best point found
            # using Nelder-Mead
            from scipy.optimize import minimize
            
            # Define a counter for the second phase
            local_state = {
                'count': 0,
                'best_x': state['best_x'].copy(),
                'best_f': state['best_f']
            }
            
            def wrapped_func_local(x):
                if time.time() > state['deadline']:
                    return 1e10
                local_state['count'] += 1
                val = func(x)
                if val < local_state['best_f']:
                    local_state['best_f'] = val
                    local_state['best_x'] = x.copy()
                return val

            try:
                # Nelder-Mead is fast and good for local refinement
                # maxiter=100 is usually enough for dim=5
                res_local = minimize(
                    wrapped_func_local, 
                    state['best_x'], 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_local.fun < state['best_f']:
                    state['best_f'] = res_local.fun
                    state['best_x'] = res_local.x
            except Exception:
                pass

        # Final check: ensure we return a valid point
        # If no function calls were made (unlikely), evaluate center
        if state['count'] == 0:
            x0 = np.zeros(dim)
            state['best_x'] = x0
            state['best_f'] = func(x0)
            state['count'] = 1

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
