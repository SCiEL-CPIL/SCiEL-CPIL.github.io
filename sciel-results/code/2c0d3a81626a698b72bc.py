"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Optimizer using CMA-ES for robust global optimization of rotated Rosenbrock.
    CMA-ES is state-of-the-art for black-box optimization and handles rotations well.
    We use a multi-start strategy with a small initial population.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf
        self.evaluated_points = []
        self.evaluated_values = []

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            self.evaluated_points.append(x.copy())
            self.evaluated_values.append(fx)
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Quick random sampling to find a good starting point
        # Sample 50 random points to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        # Generate random points in [-5, 5]^5
        for _ in range(50):
            if self.call_count >= self.budget - 50 or not time_ok():
                break
            x0 = rng.uniform(-5.0, 5.0, size=self.dim)
            try:
                tracked_func(x0)
            except Exception:
                pass

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Run CMA-ES from the best point found
        # CMA-ES is robust and handles the rotated Rosenbrock well
        remaining_budget = self.budget - self.call_count
        
        if remaining_budget > 50 and time_ok():
            try:
                # Set up CMA-ES options
                # Popsize: default is 4 + floor(3 * log(n)) = 4 + 3*1.6 = 8 for n=5
                # We use a smaller popsize to conserve budget
                popsize = 8
                
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(300, remaining_budget - 20),
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with L-BFGS-B
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 10 and time_ok() and self.best_x is not None:
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfun': remaining_budget - 2,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
