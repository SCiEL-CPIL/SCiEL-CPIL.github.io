import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) using Nelder-Mead with adaptive restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check
            if time.time() > deadline:
                return best_f
            # Budget check
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initialize best point
        # Start at origin, then try a few random points to get a better starting point
        x0 = np.zeros(dim)
        best_f = wrapper(x0)
        best_x = x0.copy()

        # Try a few random points to find a better starting point
        rng = np.random.default_rng(42)
        for _ in range(10):
            if calls_made >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-5.0, 5.0, size=dim)
            f_rand = wrapper(x_rand)
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        # Nelder-Mead optimization with restarts
        # Nelder-Mead is derivative-free and works well for Rosenbrock
        # We use multiple restarts with adaptive step sizes
        
        # Number of restarts: use remaining budget
        # Each restart uses a fraction of the remaining budget
        remaining_budget = budget - calls_made
        
        # Try Nelder-Mead from the best point
        if remaining_budget > 20 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 200), 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass

        # If we have budget left, try another restart from a slightly perturbed best point
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline - 5.0:
            try:
                # Perturb best point slightly
                x_perturbed = best_x + rng.normal(0, 0.1, size=dim)
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                res = minimize(
                    wrapper,
                    x_perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 100), 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass

        # If we still have budget, try a few more local searches
        remaining_budget = budget - calls_made
        while remaining_budget > 10 and time.time() < deadline - 2.0:
            try:
                # Random restart with small perturbation
                step_size = max(0.01, 0.5 * (1.0 - calls_made / budget))
                x_perturbed = best_x + rng.normal(0, step_size, size=dim)
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                
                res = minimize(
                    wrapper,
                    x_perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 50), 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                break
            
            remaining_budget = budget - calls_made
            if time.time() > deadline - 2.0:
                break

        return best_x, best_f
