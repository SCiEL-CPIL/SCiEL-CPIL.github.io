"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start L-BFGS-B for global optimization of BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        budget = self.budget
        
        # Initialize with a random point
        best_x = np.random.uniform(-5, 5, dim)
        best_f = func(best_x)
        
        # Strategy: Multi-Start Local Search
        # f21 is smooth, so L-BFGS-B is effective locally.
        # We perform multiple restarts to cover the multimodal landscape.
        
        # Estimate evaluations per run to stay within budget
        # L-BFGS-B typically uses ~2-5 evals per iteration.
        # Let's aim for ~50 evaluations per run.
        # With budget 500, we can do ~10 runs.
        
        n_runs = 10
        max_iter_per_run = 20  # Conservative limit to ensure we stay under budget
        
        bounds = [(-5.0, 5.0)] * dim
        
        for i in range(n_runs):
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Choose starting point
            # 50% random global, 50% local perturbation of best
            if i == 0 or np.random.rand() < 0.5:
                x0 = np.random.uniform(-5, 5, dim)
            else:
                # Perturb the current best solution
                # Scale noise by 1.0 to allow escaping local minima
                noise = np.random.normal(0, 1.0, dim)
                x0 = best_x + noise
                x0 = np.clip(x0, -5, 5)
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_per_run,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If an exception occurs, skip this run
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
