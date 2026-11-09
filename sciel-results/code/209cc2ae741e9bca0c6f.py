"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard, plus Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Track best solution for potential polish
        best_x = x.copy()
        best_fx = fx
        budget_used = 1

        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            budget_used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            if fy < best_fx:
                best_x, best_fx = y.copy(), fy

        # Final polish: Nelder-Mead local search on the best point found
        # Use remaining budget for polish iterations
        remaining = self.budget - budget_used
        if remaining > 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Create a wrapper that clips to box and tracks calls
                polish_calls = [0]
                max_polish_calls = min(remaining, 200)

                def polish_func(z):
                    if polish_calls[0] >= max_polish_calls:
                        return best_fx
                    polish_calls[0] += 1
                    z_clipped = np.clip(z, -5, 5)
                    return func(z_clipped)

                # Nelder-Mead with small initial simplex
                initial_step = 0.1
                result = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_polish_calls,
                        'initial_simplex': np.array([
                            best_x + initial_step * e
                            for e in np.eye(self.dim)
                        ]),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun
            except Exception:
                pass  # If polish fails, keep the ES result

        return best_x, best_fx
# EVOLVE-BLOCK-END
