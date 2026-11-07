"""Rewritten optimizer using finite-difference SGD for Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """SGD with finite-difference gradients and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fd_gradient(self, func, x, eps):
        """Central finite-difference gradient estimate."""
        g = np.zeros(self.dim)
        for i in range(self.dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] = min(5.0, xp[i] + eps)
            xm[i] = max(-5.0, xm[i] - eps)
            denom = xp[i] - xm[i]
            if denom < 1e-12:
                denom = eps
            g[i] = (func(xp) - func(xm)) / denom
        return g

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def remaining():
            nonlocal calls_used
            return self.budget - calls_used

        def track(func_call, x):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            f = func_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start with gradient descent
        # We have 500 calls total. Gradient estimation costs 2*dim=10 calls per step.
        # Let's do several short gradient runs from different starts.

        # Try a few initial points
        n_starts = 4
        per_start_budget = self.budget // n_starts  # 125 calls per start

        for s in range(n_starts):
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break

            # Initialize: mix of random and center
            if s == 0:
                x = np.zeros(self.dim)
            elif s == 1:
                x = rng.uniform(-3, 3, self.dim)
            elif s == 2:
                x = rng.uniform(-5, 5, self.dim)
            else:
                x = rng.uniform(-2, 2, self.dim)

            fx = track(func, x)

            # Adaptive learning rate for SGD
            lr = 0.5
            eps = 0.01  # finite difference step
            patience = 0
            max_patience = 20

            while calls_used < self.budget and time.time() < deadline:
                # Need 2*dim calls for gradient + 1 for step = 11 calls
                if remaining() < 2 * self.dim + 2:
                    break

                # Estimate gradient
                g = np.zeros(self.dim)
                for i in range(self.dim):
                    if time.time() > deadline or calls_used >= self.budget:
                        break
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] = min(5.0, xp[i] + eps)
                    xm[i] = max(-5.0, xm[i] - eps)
                    denom = xp[i] - xm[i]
                    if denom < 1e-12:
                        denom = eps
                    fxp = track(func, xp)
                    fxm = track(func, xm)
                    g[i] = (fxp - fxm) / denom

                if time.time() > deadline or calls_used >= self.budget:
                    break

                # Check if gradient is meaningful
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-6:
                    # Near optimum, try a small random perturbation
                    x = np.clip(x + 0.1 * rng.standard_normal(self.dim), -5, 5)
                    fx = track(func, x)
                    patience += 1
                    if patience > max_patience:
                        break
                    continue

                # Adaptive step size
                step = lr / g_norm
                x_new = np.clip(x - step * g, -5, 5)
                fx_new = track(func, x_new)

                if fx_new < fx:
                    x, fx = x_new, fx_new
                    lr = min(lr * 1.1, 2.0)
                    patience = 0
                    # Shrink eps as we get closer
                    eps = max(eps * 0.9, 1e-4)
                else:
                    lr *= 0.5
                    patience += 1
                    if patience > max_patience:
                        break

            # After gradient phase, use remaining calls for local refinement
            if calls_used < self.budget and time.time() < deadline:
                # Simple local random search around best point
                sigma = 0.1
                for _ in range(50):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = track(func, y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.8

        # Final phase: use any remaining budget for fine-tuning around best
        if best_x is not None and calls_used < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            while calls_used < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = track(func, y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = max(sigma * 0.9, 1e-5)
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
