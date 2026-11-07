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

        # Phase 2: Determine gradient signs using 5e_i evaluations
        a = np.zeros(d)
        for i in range(d):
            if calls >= budget or not time_ok():
                break

            xi = np.zeros(d)
            xi[i] = 5.0
            fi = try_call(xi)
            if fi is None:
                break

            a[i] = (fi - f0) / 5.0

            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Phase 3: Evaluate the optimal corner
        if calls < budget and time_ok():
            x_corner = np.zeros(d)
            for i in range(d):
                if a[i] > 0:
                    x_corner[i] = -5.0
                elif a[i] < 0:
                    x_corner[i] = 5.0
                else:
                    x_corner[i] = 0.0
            f_corner = try_call(x_corner)
            if f_corner is not None:
                if f_corner < best_f:
                    best_f = f_corner
                    best_x = x_corner.copy()

        # Phase 4: Local polish using remaining budget
        step_sizes = [0.5, 0.25, 0.1, 0.05, 0.01]
        for step in step_sizes:
            if calls >= budget or not time_ok():
                break
            for i in range(d):
                if calls >= budget or not time_ok():
                    break
                for direction in [-1, 1]:
                    if calls >= budget or not time_ok():
                        break
                    x_try = best_x.copy()
                    x_try[i] += direction * step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    if np.allclose(x_try, best_x, atol=1e-10):
                        continue
                    f_try = try_call(x_try)
                    if f_try is not None and f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        # Fill remaining budget
        while calls < budget and time_ok():
            try_call(best_x)

        return best_x, best_f
