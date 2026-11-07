import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.calls_made = 0
        self.max_time = 295.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')

        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            if self.calls_made >= self.budget:
                return float('inf')
            self.calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                return float('inf')
            if not np.isfinite(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Global Exploration with Multi-Start Powell
        # Powell is robust to the rotated/ill-conditioned landscape of Rosenbrock.
        # We use a diverse set of starting points to cover the basin of attraction.
        
        rng = np.random.default_rng(42)
        
        # Define diverse start points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2.0,
            -np.ones(self.dim) * 2.0,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0]),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
        ]

        n_starts = len(starts)
        # Allocate ~60% of budget to exploration
        exploration_budget = int(self.budget * 0.6)
        calls_per_start = max(10, exploration_budget // n_starts)

        for x0 in starts:
            if time.time() - self.start_time > self.max_time:
                break
            if self.calls_made >= exploration_budget:
                break

            remaining_for_start = min(calls_per_start, exploration_budget - self.calls_made)
            if remaining_for_start < 5:
                break

            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining_for_start,
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
                # safe_func already updates best_x/best_f
            except Exception:
                pass

        # Phase 2: Local Exploitation with L-BFGS-B
        # L-BFGS-B is efficient for smooth functions and handles bounds well.
        # We run it from the best point found in Phase 1 and a few other good points.
        
        if self.best_x is not None:
            # Use remaining budget for polish
            polish_budget = self.budget - self.calls_made
            
            if polish_budget > 10:
                # Try L-BFGS-B from the best point
                try:
                    res = minimize(
                        safe_func,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 100,
                            'maxfev': polish_budget,
                            'ftol': 1e-15,
                            'gtol': 1e-10
                        }
                    )
                except Exception:
                    pass

                # If we still have budget, try a few other good points
                # We don't have a full archive, but we can try the other starts if they were good
                # However, safe_func tracks the global best, so any improvement is captured.
                # Let's just run one more L-BFGS-B from a random point near the best if budget allows.
                if self.calls_made < self.budget - 10:
                    remaining = self.budget - self.calls_made
                    if remaining > 10:
                        # Perturb best point slightly
                        x_perturbed = self.best_x + rng.normal(0, 0.1, self.dim)
                        x_perturbed = np.clip(x_perturbed, -5, 5)
                        try:
                            res2 = minimize(
                                safe_func,
                                x_perturbed,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * self.dim,
                                options={
                                    'maxiter': 50,
                                    'maxfev': remaining,
                                    'ftol': 1e-15,
                                    'gtol': 1e-10
                                }
                            )
                        except Exception:
                            pass

        return self.best_x, self.best_f
