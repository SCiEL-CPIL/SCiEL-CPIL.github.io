"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.stats


class Optimizer:
    """Multi-start optimization using Sobol sampling and SciPy Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Global exploration with Sobol points
        # Use fewer global points to save budget for local refinement
        n_global = min(60, budget // 5)
        
        try:
            sampler = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_samples = sampler.random(n_global)
            x_global = sobol_samples * 10.0 - 5.0
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(42)
            x_global = rng.uniform(-5, 5, (n_global, dim))

        candidates = []
        best_x = None
        best_f = np.inf

        for i in range(n_global):
            if time.time() > deadline - 20:
                break
            x = x_global[i]
            fx = func(x)
            calls_used += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using SciPy Nelder-Mead on top candidates
        # We will try to refine the top 5-10 candidates
        n_refine = min(10, len(candidates))
        
        # Estimate budget per refinement
        # Leave some buffer for safety and final check
        remaining_budget = budget - calls_used - 5 
        budget_per_refine = max(10, remaining_budget // n_refine)

        for idx in range(n_refine):
            if time.time() > deadline - 10:
                break
            if calls_used >= budget - 5:
                break
            
            fx0, x0 = candidates[idx]
            
            # Define a wrapper to count calls and handle clipping/exceptions
            # Note: func is assumed to handle clipping per contract, but we ensure x is valid
            def objective(x):
                nonlocal calls_used
                if calls_used >= budget:
                    return np.inf
                # Clip to be safe, though func should do it
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx

            # Run Nelder-Mead
            try:
                res = scipy.optimize.minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': budget_per_refine, 'xatol': 1e-4, 'fatol': 1e-4}
                )
            except Exception:
                pass

            # Update best if scipy found something better
            # Note: objective updates best_f and best_x directly

        # Phase 3: Final polish with a few random perturbations or a tight NM on the best
        if calls_used < budget - 2 and time.time() < deadline - 5:
             # Try a few random steps around the best
             rng = np.random.default_rng(123)
             for _ in range(min(10, budget - calls_used)):
                 if calls_used >= budget - 1:
                     break
                 if time.time() > deadline - 2:
                     break
                 y = best_x + 0.1 * rng.standard_normal(dim)
                 y = np.clip(y, -5, 5)
                 fy = func(y)
                 calls_used += 1
                 if fy < best_f:
                     best_f = fy
                     best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
