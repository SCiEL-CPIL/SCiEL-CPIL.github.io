"""Optimizer for BBOB f5 (Linear slope) via active learning + vertex jump."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Active learning to determine gradient, then jump to optimal vertex."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _project(self, x):
        return np.clip(x, -5.0, 5.0)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Active learning - evaluate center and 5 positive axis vertices
        # This exactly determines the gradient for a linear function.
        x0 = np.zeros(dim)
        f0 = func(x0)
        x_best, f_best = x0.copy(), f0
        calls = 1

        # Evaluate at x0 + h*e_i for each dimension to get gradient components
        # Use maximum step size to maximize potential improvement in best-value tracking
        h = 5.0
        grad = np.zeros(dim)
        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            xi = x0.copy()
            xi[i] += h
            xi = self._project(xi)
            fi = func(xi)
            calls += 1
            grad[i] = (fi - f0) / h
            if fi < f_best:
                x_best, f_best = xi.copy(), fi

        # Phase 2: Determine the optimal vertex
        # For a linear function f(x) = f0 + grad^T x, the minimum on [-5,5]^dim
        # is at the vertex where each coordinate is -5 if grad[i] > 0, else 5.
        if np.any(np.abs(grad) > 1e-15):
            x_vertex = np.zeros(dim)
            for i in range(dim):
                if grad[i] > 0:
                    x_vertex[i] = -5.0
                else:
                    x_vertex[i] = 5.0
        else:
            # Degenerate case: gradient is zero, function is constant
            x_vertex = x0.copy()

        # Phase 3: Evaluate the vertex
        if calls < budget and time_ok():
            fv = func(x_vertex)
            calls += 1
            if fv < f_best:
                x_best, f_best = x_vertex.copy(), fv

        # Phase 4: Use remaining budget to re-evaluate the best point
        # This ensures b_i stays at the minimum value for the rest of the trajectory
        while calls < budget and time_ok():
            fy = func(x_best)
            calls += 1
            if fy < f_best:
                f_best = fy

        return x_best, f_best
# EVOLVE-BLOCK-END
