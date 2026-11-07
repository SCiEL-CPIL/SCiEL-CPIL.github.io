"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid scout + Powell optimizer for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Scout with several starting points to find a good basin
        # Rosenbrock has a valley; the minimum is near (1,...,1) in standard form,
        # but hidden shift/rotation moves it. Try multiple starting points.
        rng = np.random.default_rng(42)

        # Candidate starting points: origin, nominal min, and random points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.array([1.0] * self.dim),
        ]
        # Add some random starts
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_val = wrapped(s_clipped)

        # Phase 2: Run Powell's method from the best scout point
        if best_x is not None:
            # Powell's method with tight tolerances
            result = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': max(1, self.budget - call_count),
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                }
            )
            # Update best
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

            # Phase 3: If we have budget remaining, try Nelder-Mead from the best
            if call_count < self.budget and time.time() < deadline:
                result2 = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': max(1, self.budget - call_count),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                    }
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5.0, 5.0)

            # Phase 4: Final polish with Powell from the very best
            if call_count < self.budget and time.time() < deadline:
                result3 = minimize(
                    wrapped,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'maxfev': max(1, self.budget - call_count),
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
                if result3.fun < best_f:
                    best_f = result3.fun
                    best_x = np.clip(result3.x, -5.0, 5.0)

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
