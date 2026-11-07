"""Combined optimizer: multi-restart L-BFGS-B + Nelder-Mead polish for BBOB-style f19."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B + Nelder-Mead optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-restart L-BFGS-B from diverse starting points
        # L-BFGS-B uses gradient information (via finite differences) to converge
        # much faster to local minima than derivative-free methods for smooth functions.
        num_restarts = 8
        calls_per_restart = max(1, (budget - 80) // num_restarts)  # Reserve 80 for polish

        # Generate diverse starting points: center + random
        start_points = [np.zeros(n)]
        for _ in range(num_restarts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, size=n))
        rng.shuffle(start_points)

        for r, x0 in enumerate(start_points):
            if calls >= budget or time.time() >= deadline:
                break

            x0 = np.clip(x0, -5.0, 5.0)

            try:
                def wrapped_lbfgsb(x):
                    if time.time() > deadline:
                        raise ValueError("Time limit exceeded")
                    return evaluate(x)

                result = minimize(
                    wrapped_lbfgsb,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': calls_per_restart,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from the best point found
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                x0 = best_x.copy()

                def wrapped_nm(x):
                    if time.time() > deadline:
                        raise ValueError("Time limit exceeded")
                    return evaluate(x)

                result = minimize(
                    wrapped_nm,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(50, budget - calls),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Ensure at least one call was made
        if calls == 0:
            x = np.zeros(n)
            evaluate(x)

        return best_x, best_f
