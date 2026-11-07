"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds


class Optimizer:
    """
    Hybrid optimizer:
    1. Brief random walk to find a reasonable starting point.
    2. L-BFGS-B for smooth, fast convergence in the valley.
    3. Nelder-Mead for final polishing if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = np.zeros(dim)
        best_fx = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            if time.time() > deadline:
                return best_fx
            x = np.asarray(x, dtype=float)
            # Ensure finite and within bounds (though scipy should handle bounds)
            if not np.all(np.isfinite(x)):
                return 1e10
            fx = float(func(x))
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random Walk (5% of budget)
        # This helps escape any potential bad initialization and provides a diverse start.
        walk_budget = max(10, int(budget * 0.05))
        x = rng.uniform(-4, 4, dim)
        fx = wrapped_func(x)
        
        sigma = 1.0
        for i in range(walk_budget - 1):
            if calls >= budget or time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = wrapped_func(y)
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.9
                sigma = max(sigma, 1e-6)

        # Phase 2: L-BFGS-B
        # Use the best point found so far as the starting point for gradient-based optimization.
        # L-BFGS-B is effective for smooth functions like Bent Cigar.
        # We give it a generous budget but let it stop early if it converges.
        # maxiter is set high, but maxfun (calls) is the hard limit.
        # Note: scipy minimize counts function evaluations.
        # We need to be careful with the call count.
        
        remaining_budget = budget - calls
        if remaining_budget > 50 and time.time() < deadline:
            try:
                bounds = [(-5.0, 5.0)] * dim
                # L-BFGS-B parameters
                # maxfun is the maximum number of function calls.
                # We leave some buffer for Nelder-Mead.
                lbfgs_budget = int(remaining_budget * 0.7)
                
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': lbfgs_budget, 'gtol': 1e-8}
                )
                
                # Update best if scipy found something better (it should via wrapped_func)
                # wrapped_func already updates best_x/best_fx
            
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        # If we still have budget and time, use Nelder-Mead for local refinement.
        # Nelder-Mead doesn't use gradients, so it can handle any slight non-smoothness or noise.
        remaining_budget = budget - calls
        if remaining_budget > 20 and time.time() < deadline:
            try:
                nm_budget = remaining_budget
                # Nelder-Mead options
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': nm_budget, 'xatol': 1e-6, 'fatol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
