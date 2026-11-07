"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer with finite-difference gradient and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = [0]
        deadline = time.time() + 270.0  # 300s limit, keep 30s margin

        def safe_func(x):
            if calls[0] >= budget:
                return float('inf')
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return float('inf')
            return val

        # Finite difference gradient
        def gradient(x, fval, h):
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                xp = np.clip(xp, -5.0, 5.0)
                xm = np.clip(xm, -5.0, 5.0)
                grad[i] = (safe_func(xp) - safe_func(xm)) / (2.0 * h)
            return grad

        # Backtracking line search
        def line_search(x, fval, direction, step_size):
            alpha = step_size
            for _ in range(20):
                x_new = np.clip(x + alpha * direction, -5.0, 5.0)
                f_new = safe_func(x_new)
                if f_new < fval:
                    return x_new, f_new, alpha
                alpha *= 0.5
                if alpha < 1e-12:
                    break
            return x, fval, 0.0

        # Single gradient descent run
        def run_gradient_descent(x0, f0, budget_remaining):
            x = x0.copy()
            fval = f0
            h = 1e-4
            step_size = 0.5
            last_improve = 0
            patience = 100

            while budget_remaining > 0 and calls[0] < budget:
                if time.time() > deadline:
                    break

                grad = gradient(x, fval, h)
                grad_norm = np.linalg.norm(grad)

                if grad_norm < 1e-8:
                    # Try reducing h and continuing
                    h *= 0.1
                    if h < 1e-10:
                        break
                    continue

                direction = -grad / grad_norm
                x_new, f_new, alpha_used = line_search(x, fval, direction, step_size)

                if f_new < fval:
                    x, fval = x_new, f_new
                    step_size = max(alpha_used * 1.5, 0.01)
                    h = max(h * 0.95, 1e-10)
                    last_improve = 0
                else:
                    step_size *= 0.5
                    last_improve += 1
                    if last_improve > patience:
                        break

            return x, fval

        # Initialize
        rng = np.random.default_rng(42)
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Try multiple starting points with gradient descent
        n_restarts = 8
        for r in range(n_restarts):
            if calls[0] >= budget or time.time() > deadline:
                break

            if r == 0:
                x0 = np.zeros(dim)
            elif r == 1:
                x0 = np.ones(dim) * 3.0
            elif r == 2:
                x0 = np.ones(dim) * (-3.0)
            else:
                x0 = rng.uniform(-4.0, 4.0, dim)

            f0 = safe_func(x0)
            if not np.isfinite(f0):
                continue

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run gradient descent from this point
            remaining = budget - calls[0]
            if remaining < 10:
                break

            x_res, f_res = run_gradient_descent(x0, f0, remaining)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # If we have budget left, do a final refinement
        if calls[0] < budget - 5:
            x_res, f_res = run_gradient_descent(best_x, best_f, budget - calls[0])
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
