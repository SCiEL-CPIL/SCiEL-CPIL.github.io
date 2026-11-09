"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer using Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap func to count evaluations
        eval_count = [0]
        def wrapped_func(x):
            eval_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts with Nelder-Mead
        # First, do a coarse random search to find a good starting point
        n_random = min(50, budget // 10)
        for _ in range(n_random):
            if eval_count[0] >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Now use Nelder-Mead from the best point found
        # We'll use multiple restarts if we have budget left
        remaining = budget - eval_count[0]
        
        # Try Nelder-Mead with different initial simplex sizes
        simplex_sizes = [2.0, 1.0, 0.5, 0.2, 0.1]
        
        for size in simplex_sizes:
            if remaining - eval_count[0] < 10:
                break
            
            # Create initial simplex
            x0 = best_x.copy()
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += size
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
            
            # Evaluate simplex points
            fvals = []
            for s in simplex:
                if eval_count[0] >= budget:
                    break
                fvals.append(wrapped_func(s))
            
            if len(fvals) < dim + 1:
                break
            
            # Use scipy's Nelder-Mead with our wrapped function
            # Set maxiter based on remaining budget
            max_iter = max(1, (budget - eval_count[0]) // (dim + 1))
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass
            
            # If we're close to convergence, try a smaller simplex
            if eval_count[0] >= budget:
                break

        # Final polish with a few more random perturbations around best
        if eval_count[0] < budget:
            for _ in range(20):
                if eval_count[0] >= budget:
                    break
                perturbation = rng.normal(0, 0.1, dim)
                x_new = np.clip(best_x + perturbation, -5, 5)
                fx = wrapped_func(x_new)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
