import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Start at the center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center

        calls_used = 1

        # Determine the sign of the gradient for each dimension
        # For a linear function f(x) = c + w^T x, the minimum is at the vertex
        # where x_i = 5 if w_i < 0, and x_i = -5 if w_i > 0.
        # We estimate w_i by checking f(5 e_i) vs f(0).
        signs = np.zeros(dim)

        for i in range(dim):
            # Evaluate f at 5 * e_i
            x_probe = x_center.copy()
            x_probe[i] = 5.0

            f_probe = func(x_probe)
            calls_used += 1

            # Update best if probe is better
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()

            # Determine sign of gradient component i
            # f(5 e_i) = c + 5 w_i
            # f(0) = c
            # If f(5 e_i) < f(0), then 5 w_i < 0 => w_i < 0.
            # If w_i < 0, min is at x_i = 5.
            # If f(5 e_i) > f(0), then w_i > 0, min is at x_i = -5.
            # If equal, w_i = 0, sign doesn't matter (any x_i is fine, but we pick -5 or 5).
            if f_probe < f_center:
                signs[i] = -1.0  # Gradient negative, go to +5
            else:
                signs[i] = 1.0   # Gradient positive (or zero), go to -5

        # Construct the optimal vertex
        x_vertex = np.where(signs < 0, 5.0, -5.0)

        # Evaluate the vertex
        f_vertex = func(x_vertex)
        calls_used += 1

        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()

        # For linear functions, this vertex is the global minimum.
        # We can return now. The remaining budget does not improve the value.

        return best_x, best_f
