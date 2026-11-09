import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Split budget: 70% for Global Search (DE), 30% for Local Polish
        de_budget = int(total_budget * 0.7)
        polish_budget = total_budget - de_budget

        # Calculate maxiter for DE
        # popsize=2 means population size is 2 * dim
        pop_factor = 2
        pop_size = pop_factor * n
        # Evals = pop_size + pop_size * maxiter
        # maxiter = (de_budget - pop_size) / pop_size
        maxiter_de = max(1, int((de_budget - pop_size) / pop_size))

        # Run Differential Evolution
        # bounds: list of tuples
        bounds = [(-5, 5)] * n
        seed = 42
        
        try:
            # polish=False to avoid extra evaluations inside DE
            res_de = differential_evolution(
                wrapped_func, 
                bounds=bounds, 
                maxiter=maxiter_de,
                popsize=pop_factor,
                seed=seed,
                polish=False,
                tol=1e-8
            )
        except (TimeoutError, ValueError) as e:
            # If DE fails or exceeds budget/time, proceed to polish with best found so far
            pass
        except Exception as e:
            # Catch other errors to ensure we return something
            pass

        # Now polish the best result found
        # Use remaining budget for local optimization
        # Powell is good for ridges
        # We need to limit the number of function calls in minimize
        # minimize doesn't have max_evals, so we rely on the wrapper raising an exception
        
        # Reset calls for polish? No, the wrapper counts total.
        # We just call minimize. If it exceeds budget, it will raise ValueError.
        
        x0 = best_x
        if np.isinf(best_f):
             # If no valid point found, pick a random one
             x0 = np.random.uniform(-5, 5, n)
             best_f = wrapped_func(x0)

        try:
            res_local = minimize(
                wrapped_func, 
                x0, 
                method='Powell',
                bounds=bounds,
                options={'maxiter': 100} # Arbitrary high, will be cut by budget
            )
            # minimize returns a result object. 
            # But wrapped_func updates best_x and best_f.
            # We should check if res_local.fun is better
            if res_local.fun < best_f:
                best_f = res_local.fun
                best_x = np.clip(res_local.x, -5, 5)
        except (TimeoutError, ValueError) as e:
            # Budget or time exceeded during polish
            pass
        except Exception as e:
            pass

        return best_x, best_f
