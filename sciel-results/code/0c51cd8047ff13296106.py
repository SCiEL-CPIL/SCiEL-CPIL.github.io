"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart Nelder-Mead optimizer for f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        rng = np.random.RandomState(42)

        # Generate diverse initial points using stratified sampling
        n_initial = 15
        X_samples = []
        f_samples = []

        # Center
        x0 = np.zeros(d)
        fx0 = safe_func(x0)
        X_samples.append(x0.copy())
        f_samples.append(fx0)

        # Random points
        for _ in range(n_initial - 1):
            x = rng.uniform(-5, 5, size=d)
            fx = safe_func(x)
            X_samples.append(x.copy())
            f_samples.append(fx)

        X_samples = np.array(X_samples)
        f_samples = np.array(f_samples)

        # Multi-restart Nelder-Mead with diverse starting points
        # Use best points from initial sampling plus some random restarts
        start_points = []

        # Add best initial points
        top_k = min(5, len(X_samples))
        indices = np.argsort(f_samples)[:top_k]
        for i in indices:
            start_points.append(X_samples[i].copy())

        # Add diverse random points
        for _ in range(5):
            start_points.append(rng.uniform(-5, 5, size=d))

        # Run Nelder-Mead from each starting point
        for x0 in start_points:
            if not time_ok() or calls >= budget - 10:
                break

            # Perturb the starting point slightly for diversity
            x_init = x0 + rng.normal(0, 0.1, size=d)
            x_init = np.clip(x_init, -5, 5)

            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)

            try:
                res = minimize(
                    local_func,
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, do a final refinement with L-BFGS-B
        if time_ok() and calls < budget - 5:
            bounds = [(-5, 5)] * d

            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)

            try:
                res = minimize(
                    local_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
