"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: Sobol exploration + Multi-start Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse search using Sobol sequence
        # We use a portion of the budget for global exploration
        n_explore = min(100, budget // 5)
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # We need to generate enough points. Sobol requires power of 2 for best distribution, 
        # but we can just generate n_explore points.
        sobol_points = sampler.random(n_explore)
        
        # Map from [0, 1]^dim to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate all Sobol points
        for x in sobol_points:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = func(x)
            calls_used += 1
            update_best(x, fx)

        # If no points were evaluated (shouldn't happen), fallback to random
        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls_used += 1
            update_best(x, fx)

        # Phase 2: Local refinement using Nelder-Mead
        # We will try to refine the best point found so far.
        # We also keep a list of top candidates from Phase 1 to restart if needed.
        
        # Sort all evaluated points? We didn't store them all. 
        # Let's store top few from Phase 1.
        # Re-evaluating top candidates is cheap if we just keep them.
        # But we didn't keep them. Let's just use the best one.
        # To be more robust, let's do a few random restarts in Phase 2 as well?
        # No, let's stick to refining the best point.

        # Configure Nelder-Mead
        # maxiter needs to be set such that we don't exceed budget.
        # Each iteration of NM in dim=5 uses roughly dim+1 = 6 evaluations? 
        # Actually, NM simplex has dim+1 vertices. Each step involves 1 new evaluation.
        # So maxiter ~ budget_remaining / 2 is safe.
        
        remaining_budget = budget - calls_used
        if remaining_budget <= 10:
            return best_x, best_f

        # We will run NM in a loop, restarting if it fails or if we find a better point
        # and have budget left.
        
        # Initial simplex: best_x + small perturbations
        # Standard NM initialization
        initial_simplex = [best_x]
        for i in range(dim):
            x_perturbed = best_x.copy()
            # Perturb each dimension
            x_perturbed[i] += 0.1 if x_perturbed[i] < 0 else -0.1
            # Ensure bounds
            x_perturbed = np.clip(x_perturbed, -5, 5)
            initial_simplex.append(x_perturbed)
        
        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            nonlocal calls_used
            if time.time() > deadline or calls_used >= budget:
                raise Exception("Budget or time exceeded")
            fx = func(x)
            calls_used += 1
            update_best(x, fx)
            return fx

        try:
            res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': remaining_budget // 2, 'maxfev': remaining_budget})
        except Exception:
            pass

        # If we have significant budget left, we might try another local search from a different point?
        # For now, single start from best global point is the structural change from the broken CMA-ES.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
