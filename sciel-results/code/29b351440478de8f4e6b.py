import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        d = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def try_call(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # Phase 1: Evaluate center first
        x0 = np.zeros(d)
        f0 = try_call(x0)
        if f0 is None:
            return x0, 0.0

        best_x = x0.copy()
        best_f = f0

        # Phase 2: For each dimension, evaluate 5e_i, compute a_i, and evaluate partial x_opt
        a = np.zeros(d)
        for i in range(d):
            if calls >= budget or not time_ok():
                break

            # Evaluate 5e_i
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = try_call(xi)
            if fi is None:
                break

            a[i] = (fi - f0) / 5.0

            # Update best from 5e_i
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

            # Evaluate partial x_opt
            partial = np.zeros(d)
            for j in range(i + 1):
                if a[j] > 0:
                    partial[j] = -5.0
                elif a[j] < 0:
                    partial[j] = 5.0
                else:
                    partial[j] = 0.0
            fp = try_call(partial)
            if fp is not None:
                if fp < best_f:
                    best_f = fp
                    best_x = partial.copy()

        # Phase 3: Evaluate full x_opt
        if calls < budget and time_ok():
            x_opt = np.zeros(d)
            for i in range(d):
                if a[i] > 0:
                    x_opt[i] = -5.0
                elif a[i] < 0:
                    x_opt[i] = 5.0
                else:
                    x_opt[i] = 0.0
            f_opt_val = try_call(x_opt)
            if f_opt_val is not None:
                if f_opt_val < best_f:
                    best_f = f_opt_val
                    best_x = x_opt.copy()

        # Fill remaining budget
        while calls < budget and time_ok():
            try_call(best_x)

        return best_x, best_f
