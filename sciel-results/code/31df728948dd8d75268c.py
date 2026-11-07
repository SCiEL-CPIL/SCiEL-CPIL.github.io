"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step size and strict wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Counting wrapper
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        def remaining_time():
            return deadline - time.time()

        # Generate diverse starting points
        starts = [np.zeros(dim)]
        for _ in range(6):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        for _ in range(3):
            starts.append(rng.uniform(-5.0, 5.0, dim) * rng.choice([-1.0, 1.0], size=dim))

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        evaluated_starts = []
        for x0 in starts:
            if call_count >= budget or remaining_time() < 30.0:
                break
            f0 = counted_func(x0)
            evaluated_starts.append((x0.copy(), f0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)
            evaluated_starts.append((best_x.copy(), best_f))

        # Sort starts by fitness (best first)
        evaluated_starts.sort(key=lambda t: t[1])

        # Run Nelder-Mead from each start point, starting with the best
        for x0, f0 in evaluated_starts:
            if call_count >= budget or remaining_time() < 10.0:
                break

            # Nelder-Mead options
            options = {
                'maxiter': 200,
                'maxfev': 100,
                'xatol': 1e-8,
                'fatol': 1e-10,
            }

            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='Nelder-Mead',
                    options=options
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # After NM, do a (1+1)-ES refinement for fine-tuning
            if call_count < budget and remaining_time() > 5.0:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5

                for _ in range(50):
                    if call_count >= budget or remaining_time() < 5.0:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = counted_func(y)

                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                        if sigma < 1e-10:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
