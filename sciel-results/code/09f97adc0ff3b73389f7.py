"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit linear structure of f5: estimate gradient, jump to boundary, then polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        x = np.zeros(self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        calls = 1

        def time_ok():
            return time.time() < deadline

        # Phase 1: Estimate gradient via central differences
        h = 0.01
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            if not time_ok():
                break
            xp = x.copy()
            xp[i] += h
            xm = x.copy()
            xm[i] -= h
            f_plus = func(xp)
            calls += 1
            f_minus = func(xm)
            calls += 1
            grad[i] = (f_plus - f_minus) / (2 * h)

        # Phase 2: Jump to the boundary point that minimizes the linear approximation
        # For a linear function f(x) ≈ f0 + grad·(x - x0), the minimum over [-5,5]^d
        # is at each coordinate: -5 if grad[i] > 0, +5 if grad[i] < 0
        x_jump = np.where(grad > 0, -5.0, 5.0)
        # If grad is near zero for some coordinate, keep it at 0
        x_jump[np.abs(grad) < 1e-6] = 0.0
        fx_jump = func(x_jump)
        calls += 1
        if fx_jump < best_fx:
            best_x, best_fx = x_jump.copy(), fx_jump

        # Phase 3: Local polish from the best point using a small-step (1+1)-ES
        x_cur = best_x.copy()
        fx_cur = best_fx
        sigma = 0.1

        while calls < self.budget and time_ok():
            # Estimate local gradient more carefully
            h_local = 0.001
            grad_local = np.zeros(self.dim)
            for i in range(self.dim):
                if not time_ok() or calls >= self.budget:
                    break
                xp = x_cur.copy()
                xp[i] = np.clip(xp[i] + h_local, -5, 5)
                xm = x_cur.copy()
                xm[i] = np.clip(xm[i] - h_local, -5, 5)
                f_plus = func(xp)
                calls += 1
                f_minus = func(xm)
                calls += 1
                grad_local[i] = (f_plus - f_minus) / (2 * h_local)

            # If gradient is negligible, we're close to optimum
            if np.max(np.abs(grad_local)) < 1e-8:
                break

            # Take a gradient descent step
            step_size = 0.5
            x_new = x_cur - step_size * grad_local / (np.max(np.abs(grad_local)) + 1e-10)
            x_new = np.clip(x_new, -5, 5)
            fx_new = func(x_new)
            calls += 1

            if fx_new < fx_cur:
                x_cur = x_new
                fx_cur = fx_new
                if fx_cur < best_fx:
                    best_x, best_fx = x_cur.copy(), fx_cur
                step_size = min(step_size * 1.2, 2.0)
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
