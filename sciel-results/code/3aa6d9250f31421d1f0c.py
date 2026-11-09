"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Combined Differential Evolution and Nelder-Mead for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                # Return a large value to stop optimization, 
                # but we must be careful not to raise exceptions that kill the process
                # if the caller expects a float. 
                # However, DE and NM handle exceptions by stopping.
                # Raising an exception is safer to break loops immediately.
                raise RuntimeError("Budget or Time Exceeded")
            
            x = np.asarray(x, dtype=float)
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            
            try:
                f_val = float(func(x_clipped))
            except Exception:
                f_val = float('inf')
            
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # DE is robust to ill-conditioning and multimodality.
        # We allocate ~60% of the budget to DE.
        de_budget = int(budget * 0.6)
        
        # Determine popsize and maxiter to stay within de_budget
        # DE uses popsize evaluations per generation.
        popsize = 15
        max_iter_de = max(1, de_budget // popsize)
        
        try:
            # bounds: [-5, 5] for all dimensions
            bounds = [(-5.0, 5.0)] * dim
            
            # Use a seed for reproducibility if needed, but random is fine
            # strategy='best1bin' is standard
            res_de = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=max_iter_de, 
                popsize=popsize,
                seed=42,
                tol=1e-8,
                polish=False  # We will do our own polish
            )
            
            # Update best if DE found something better
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x
        except Exception:
            pass

        # Phase 2: Local Refinement with Nelder-Mead
        # Use the remaining budget for NM to converge to the local minimum.
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            # NM needs at least dim+1 calls for initial simplex
            # Plus iterations.
            # Let's allow NM to use up to remaining_budget calls.
            # We need to create a new wrapper or reuse the one?
            # The wrapper 'wrapped_func' increments 'calls'. 
            # We can just call minimize with the same wrapped_func.
            
            # Initial guess is the best found by DE
            x0 = best_x.copy()
            
            # Simplex initialization
            # Use a small step size since we are likely close to the minimum
            delta = 0.1
            simplex = [x0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # NM options
            # maxiter is a loose bound, the real limit is 'calls' in wrapped_func
            options = {
                'maxiter': 1000, # Will be cut off by budget
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True,
                'initial_simplex': simplex
            }
            
            try:
                minimize(wrapped_func, x0, method='Nelder-Mead', options=options)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
