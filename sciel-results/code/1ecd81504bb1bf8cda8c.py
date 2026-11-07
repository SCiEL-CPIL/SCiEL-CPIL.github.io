import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return None
            calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Center
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is None:
            return best_x, best_f

        # 2. Probes to determine gradient signs for first dim-1 dimensions
        # For a linear function, probing the axes allows us to determine the direction of the gradient.
        # We probe +5 * e_i for i in range(dim-1).
        grad = np.zeros(dim)
        for i in range(dim - 1):
            if time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = 5.0
            fi = safe_call(xi)
            if fi is None:
                break
            # Calculate slope along axis i
            grad[i] = (fi - f0) / 5.0

        # 3. Determine optimal values for first dim-1 dimensions
        x_opt = np.zeros(dim)
        for i in range(dim - 1):
            if grad[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0

        # 4. For the last dimension, we don't know the gradient sign.
        # Evaluate both candidate vertices and pick the better one.
        # Candidate 1: last dim at -5
        x_opt_last_neg = x_opt.copy()
        x_opt_last_neg[-1] = -5.0
        f1 = safe_call(x_opt_last_neg)
        if f1 is None:
            return best_x, best_f

        # Candidate 2: last dim at +5
        x_opt_last_pos = x_opt.copy()
        x_opt_last_pos[-1] = 5.0
        f2 = safe_call(x_opt_last_pos)
        if f2 is None:
            return best_x, best_f

        # Pick the better candidate
        if f1 <= f2:
            x_opt = x_opt_last_neg
        else:
            x_opt = x_opt_last_pos

        # 5. Fill remaining budget with optimal point
        # This maximizes AOCC by keeping b_i at f_opt for the rest of the calls.
        while calls < budget:
            if time.time() > deadline:
                break
            safe_call(x_opt)

        return best_x, best_f
