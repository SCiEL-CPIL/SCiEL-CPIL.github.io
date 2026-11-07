"""Combined optimizer: Differential Evolution + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """
    Differential Evolution + L-BFGS-B optimizer.
    
    Strategy:
    1. Phase 1: Global exploration using Differential Evolution (DE).
       DE is robust to multimodality and rotation, maintaining a population
       that explores the landscape efficiently.
    2. Phase 2: Local refinement using L-BFGS-B starting from the best
       point found by DE. This exploits the smoothness of the function
       to achieve high precision quickly.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
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
        
        bounds = [(-5, 5)] * self.dim
        
        # Phase 1: Differential Evolution
        # Use a small population to fit within budget
        # popsize * maxiter <= ~400
        popsize = 10
        maxiter = 35
        
        try:
            # DE will use its own random generator if seed is provided
            # We use a fixed seed for reproducibility
            result_de = differential_evolution(
                evaluate,
                bounds,
                popsize=popsize,
                maxiter=maxiter,
                seed=42,
                tol=1e-8,
                polish=False  # We do our own polish
            )
            if result_de.fun < best_f:
                best_f = result_de.fun
                best_x = result_de.x
        except Exception:
            pass
        
        # Phase 2: L-BFGS-B Polish
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and best_x is not None:
            if time.time() < deadline:
                try:
                    result_lbfgs = minimize(
                        evaluate,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': min(50, remaining_budget - 5),
                            'ftol': 1e-12,
                            'gtol': 1e-8
                        }
                    )
                    if result_lbfgs.fun < best_f:
                        best_f = result_lbfgs.fun
                        best_x = result_lbfgs.x
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
