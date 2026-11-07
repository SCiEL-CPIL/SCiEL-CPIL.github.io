"""Optimizer for BBOB-style f9 (Rosenbrock rotated) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Finite-difference gradient descent with adaptive step size and multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fd_gradient(self, func, x, h, n_calls_ref):
        """Central finite-difference gradient. Returns (grad, calls_used)."""
        grad = np.zeros(self.dim)
        calls = 0
        for i in range(self.dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            fp = func(xp)
            fm = func(xm)
            calls += 2
            grad[i] = (fp - fm) / (2.0 * h)
        return grad, calls

    def _gd_phase(self, func, x0, fx0, n_calls, deadline, budget, max_iter, h0):
        """Run adaptive gradient descent from x0. Returns (x_best, fx_best, n_calls)."""
        x = x0.copy()
        fx = fx0
        h = h0
        alpha = 1.0
        stall = 0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if n_calls >= budget:
                break
            grad, c = self._fd_gradient(func, x, h, n_calls)
            n_calls += c
            if n_calls >= budget:
                break
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            direction = -grad / gnorm
            # Adaptive step: try alpha, backtrack if no improvement
            step = alpha * h
            y = np.clip(x + step * direction, -5.0, 5.0)
            fy = func(y)
            n_calls += 1
            if fy < fx:
                x, fx = y, fy
                stall = 0
                alpha = min(alpha * 1.5, 5.0)
                h = min(h * 1.1, 1.0)
            else:
                stall += 1
                alpha *= 0.5
                if stall > 10:
                    # Try a larger exploratory step
                    step = 2.0 * h
                    y = np.clip(x + step * direction, -5.0, 5.0)
                    fy = func(y)
                    n_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        stall = 0
                        alpha = 1.0
                    else:
                        break
        return x, fx, n_calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s / 330s kill
        budget = self.budget
        dim = self.dim
        n_calls = 0

        # Initial point: center is a strong prior for Rosenbrock
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        n_calls += 1

        # Phase 1: gradient descent from center
        x_best, fx_best, n_calls = self._gd_phase(
            func, x_best, fx_best, n_calls, deadline, budget,
            max_iter=200, h0=0.05
        )

        # Phase 2: multi-start refinement if budget remains
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
        ]
        # Add a few random starts
        for _ in range(5):
            starts.append(rng.uniform(-3, 3, dim))

        h0 = 0.03
        for start in starts:
            if time.time() > deadline:
                break
            if n_calls >= budget - 10:
                break
            x0 = np.clip(start, -5.0, 5.0)
            fx0 = func(x0)
            n_calls += 1
            if n_calls >= budget:
                break
            if fx0 < fx_best:
                x_best, fx_best = x0, fx0
            # Run GD from this start
            x, fx, n_calls = self._gd_phase(
                func, x0, fx0, n_calls, deadline, budget,
                max_iter=80, h0=h0
            )
            if fx < fx_best:
                x_best, fx_best = x, fx

        # Phase 3: final fine-grained polish if budget remains
        if n_calls < budget - 10 and time.time() < deadline:
            x, fx, n_calls = self._gd_phase(
                func, x_best, fx_best, n_calls, deadline, budget,
                max_iter=50, h0=0.01
            )
            if fx < fx_best:
                x_best, fx_best = x, fx

        return x_best, fx_best
# EVOLVE-BLOCK-END
