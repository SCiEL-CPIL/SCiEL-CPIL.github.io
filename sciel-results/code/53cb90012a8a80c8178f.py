"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, gradient estimation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        call_count = 1

        sigma = 1.5

        def try_update(y, fy):
            nonlocal best_x, best_f, call_count
            call_count += 1
            if fy < best_f:
                best_x = y.copy()
                best_f = fy
            return fy

        for i in range(budget - 1):
            # Time guard
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Periodically try gradient-based refinement
            if call_count < budget - dim * 2 and i % 10 == 5:
                # Estimate gradient at current best point
                eps = max(sigma * 0.1, 1e-4)
                grad = np.zeros(dim)
                for j in range(dim):
                    if call_count >= budget - 1:
                        break
                    xp = best_x.copy()
                    xm = best_x.copy()
                    xp[j] = np.clip(xp[j] + eps, -5, 5)
                    xm[j] = np.clip(xm[j] - eps, -5, 5)
                    fp = func(xp)
                    fm = func(xm)
                    call_count += 2
                    grad[j] = (fp - fm) / (2 * eps)

                if call_count < budget - 1:
                    # Take a gradient step
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 1e-10:
                        step_size = min(sigma * 0.5, 1.0)
                        x_new = np.clip(best_x - step_size * grad / grad_norm, -5, 5)
                        f_new = func(x_new)
                        call_count += 1
                        if f_new < best_f:
                            best_x = x_new
                            best_f = f_new
                            x = x_new
                            fx = f_new
                            sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
