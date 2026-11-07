"""Multi-restart L-BFGS-B + Nelder-Mead optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """Multi-restart gradient-based optimization with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

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
        # Use a small number of restarts to explore the rotated landscape.
        num_restarts = 10
        calls_per_restart = max(1, (budget - 80) // num_restarts)  # Reserve 80 for polish

        for r in range(num_restarts):
            if calls >= budget or time.time() >= deadline:
                break

            # Generate diverse starting point
            if r == 0:
                x0 = np.zeros(n)
            elif r == 1:
                x0 = np.ones(n) * 2.0
            elif r == 2:
                x0 = np.ones(n) * -2.0
            else:
                x0 = rng.uniform(-4.5, 4.5, size=n)

            x0 = np.clip(x0, -5.0, 5.0)

            # Try L-BFGS-B from this starting point
            try:
                from scipy.optimize import minimize

                # Wrapper for L-BFGS-B
                def wrapped_lbfgsb(x):
                    return evaluate(x)

                result = minimize(
                    wrapped_lbfgsb,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': calls_per_restart,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                    }
                )

            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from the best point found
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize

                x0 = best_x.copy()

                def wrapped_nm(x):
                    return evaluate(x)

                # Nelder-Mead with tight tolerances
                result = minimize(
                    wrapped_nm,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(50, budget - calls),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )

            except Exception:
                pass

        # Phase 3: Final high-precision polish with L-BFGS-B from the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize

                x0 = best_x.copy()

                def wrapped_final_lbfgsb(x):
                    return evaluate(x)

                result = minimize(
                    wrapped_final_lbfgsb,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max(20, budget - calls),
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )

            except Exception:
                pass

        return best_x, best_f
