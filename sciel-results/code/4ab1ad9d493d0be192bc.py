import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        
        d = self.dim
        bounds = [(-5.0, 5.0) for _ in range(d)]
        
        # State tracking
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return None
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        def make_wrapper(func_obj):
            """Create a wrapper that respects budget and deadline."""
            def wrapper(x):
                if calls >= self.budget or time.time() > deadline - 5:
                    return 1e10
                val = func_obj(x)
                if val is None:
                    return 1e10
                return val
            return wrapper

        # Phase 1: Global Search via Random Sampling
        # The Bent Cigar function has a very narrow valley. 
        # We need to find the valley's direction first.
        # Random sampling provides a global perspective.
        n_samples = 40
        rng = np.random.default_rng()
        
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)
        
        if best_x is None:
            best_x = np.zeros(d)
            evaluate(best_x)
            
        # Phase 2: Local Refinement via L-BFGS-B
        # Once in the basin of attraction (valley), L-BFGS-B is extremely efficient
        # because it handles the ill-conditioning better than simple gradient descent
        # and requires fewer function evaluations per step of progress along the valley.
        
        if calls < self.budget - 10 and time.time() < deadline - 10:
            lbfgsb_wrapper = make_wrapper(evaluate)
            
            # Start from the best point found in Phase 1
            x_start = best_x
            
            try:
                # L-BFGS-B with bounds
                # maxfev is set high, but the wrapper enforces the hard budget
                result = minimize(
                    lbfgsb_wrapper,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                # The wrapper updates best_x/best_f internally
            except Exception:
                pass

        # Phase 3: Final Polish via Nelder-Mead
        # Nelder-Mead is derivative-free and robust. It can help if L-BFGS-B gets stuck
        # due to numerical issues or if the landscape has slight non-smoothness (though BBOB is smooth).
        # It uses ~d+1 evaluations per iteration, so it's expensive but good for final convergence.
        
        if calls < self.budget - 10 and time.time() < deadline - 5:
            nm_wrapper = make_wrapper(evaluate)
            x_start = best_x if best_x is not None else np.zeros(d)
            
            try:
                # Nelder-Mead
                result = minimize(
                    nm_wrapper,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-12
                    }
                )
            except Exception:
                pass

        # Return the best found
        if best_x is None:
            best_x = np.zeros(d)
            best_f = 0.0
            
        return best_x, best_f
