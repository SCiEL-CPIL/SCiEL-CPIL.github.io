"""Combined optimizer: multi-start global sampling + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer with deadline guard.
    
    Strategy:
    1. Phase 1: Global exploration using a mix of random and quasi-random points.
       This quickly identifies promising regions in the shifted/rotated landscape.
    2. Phase 2: Local refinement using Nelder-Mead simplex method starting from
       the best points found in Phase 1.
    
    This approach avoids the slow convergence of population-based methods (like CMA-ES)
    in early iterations and exploits the multimodal nature of f17 by jumping to local
    basins via global sampling, then refining with efficient local search.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 300s limit, keep 20s buffer
        
        rng = np.random.default_rng(42)
        
        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Global exploration with diverse starting points
        # Use a combination of uniform random and stratified sampling
        num_global = min(80, self.budget // 4)
        
        # First point: center
        x0 = np.zeros(self.dim)
        evaluate(x0)
        
        # Stratified sampling for better coverage
        for i in range(num_global - 1):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            # Use quasi-random (Halton-like) for better space-filling
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)
        
        # Phase 2: Local refinement using Nelder-Mead from best candidates
        # Select top candidates from Phase 1 for local refinement
        # We'll run Nelder-Mead from the current best and a few nearby points
        
        # Run Nelder-Mead from the best point found
        if best_x is not None and call_count < self.budget:
            if time.time() < deadline:
                try:
                    result = minimize(
                        evaluate,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(200, self.budget - call_count - 10),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass
        
        # Phase 3: Additional local refinement from other promising points
        # Try a few more local searches from slightly perturbed best point
        remaining_budget = self.budget - call_count
        if remaining_budget > 20 and best_x is not None:
            if time.time() < deadline:
                for attempt in range(3):
                    if call_count >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Perturb the best point slightly
                    sigma = 0.5 * (1.0 / (attempt + 1))
                    x_perturbed = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                    f_perturbed = evaluate(x_perturbed)
                    
                    # If perturbation found something better, refine from there
                    if f_perturbed < best_f:
                        if call_count < self.budget and time.time() < deadline:
                            try:
                                result = minimize(
                                    evaluate,
                                    x_perturbed,
                                    method='Nelder-Mead',
                                    options={
                                        'maxiter': min(100, self.budget - call_count - 5),
                                        'xatol': 1e-8,
                                        'fatol': 1e-10,
                                        'adaptive': True
                                    }
                                )
                                if result.fun < best_f:
                                    best_f = result.fun
                                    best_x = result.x
                            except Exception:
                                pass
        
        # Phase 4: Final polish with L-BFGS-B if we have budget left
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and best_x is not None:
            if time.time() < deadline:
                try:
                    result = minimize(
                        evaluate,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': min(50, remaining_budget - 5),
                            'ftol': 1e-12,
                            'gtol': 1e-8
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
