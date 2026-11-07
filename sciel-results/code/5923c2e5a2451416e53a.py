"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware hybrid: random exploration + Powell local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample points in [-4, 4]^5 to find a good basin.
        # Rastrigin is multimodal, so we need to cover the space.
        # We use a mix of uniform random and a few structured points.
        n_init = 100
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                # Try 0 first, it's a strong prior for unshifted Rastrigin
                x = np.zeros(dim)
            else:
                # Sample uniformly in [-4, 4] to stay within the likely region of the minimum
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Local Refinement using Powell's method
        # Powell is derivative-free and robust for multimodal functions.
        # We run it from the best point found in Phase 1.
        
        # We will run Powell multiple times if we have budget left, restarting from perturbations
        # of the current best point to escape potential local minima or refine further.
        
        max_powell_runs = 5
        for run_idx in range(max_powell_runs):
            if calls_made >= budget or time.time() > deadline - 10:
                break
            
            # Determine starting point for this Powell run
            if run_idx == 0:
                x_start = best_x.copy()
            else:
                # Perturb the best point to escape local minima
                # The perturbation size should be small enough to stay in a similar basin
                # but large enough to potentially jump to a better one.
                # Rastrigin's local minima are spaced by ~2*pi, so a small perturbation is safe.
                sigma = 0.5 / (run_idx + 1)  # Decreasing perturbation size
                x_start = best_x + rng.normal(0, sigma, dim)
                x_start = np.clip(x_start, -5, 5)
            
            try:
                res = minimize(
                    counted_func, 
                    x_start, 
                    method='Powell', 
                    options={'maxiter': 100, 'maxfev': 100}  # Limit function evaluations per run
                )
            except Exception:
                pass

        # Phase 3: Final Polish (Optional)
        # If we have time and budget left, do a very tight Nelder-Mead or Powell run
        # to squeeze out the last bit of precision.
        if calls_made < budget and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
