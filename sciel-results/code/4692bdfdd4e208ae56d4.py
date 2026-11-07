# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """DE (SciPy) + Nelder-Mead hybrid for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget

        # Phase 1: Differential Evolution using SciPy
        # We want to use approx 80% of budget for global search
        de_budget = int(budget * 0.8)
        
        # popsize=5 means 5*dim = 25 individuals.
        # 400 budget / 25 = 16 iterations.
        # Set maxiter to 15 to be safe.
        
        def wrapped_func(x):
            if self._call_count >= de_budget:
                return np.inf
            return count_eval(x)

        bounds = [(-5, 5) for _ in range(dim)]
        
        try:
            de_result = differential_evolution(
                wrapped_func, 
                bounds, 
                popsize=5, 
                maxiter=15, 
                tol=0.01,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < self._best_f:
                self._best_f = de_result.fun
                self._best_x = de_result.x
        except Exception:
            pass

        # Phase 2: Nelder-Mead local search from best point (use remaining budget)
        remaining = budget - self._call_count
        if remaining > 10 and self._best_x is not None and time.time() < deadline:
            def wrapper(x):
                if self._call_count >= budget:
                    return self._best_f
                x_clipped = np.clip(x, -5, 5)
                return count_eval(x_clipped)

            x0 = self._best_x.copy()
            options = {
                'maxiter': min(remaining // 2, 500),
                'maxfev': min(remaining, 500),
                'xatol': 1e-8,
                'fatol': 1e-10,
            }

            try:
                result = minimize(
                    wrapper, x0, method='Nelder-Mead',
                    options=options, timeout=max(1, deadline - time.time())
                )
                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Final polish with small perturbations
        if self._call_count < budget - 5 and self._best_x is not None and time.time() < deadline:
            for _ in range(min(20, budget - self._call_count - 2)):
                if self._call_count >= budget - 2 or time.time() > deadline:
                    break
                scale = 0.05
                perturbation = rng.standard_normal(dim) * scale
                x_candidate = np.clip(self._best_x + perturbation, -5, 5)
                count_eval(x_candidate)

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
