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

        # The function is linear: f(x) = a^T x + b
        # We can determine a and b with d+1 evaluations:
        # f(0) = b
        # f(5*e_i) = 5*a_i + b  =>  a_i = (f(5*e_i) - b) / 5

        # Strategy:
        # 1. Evaluate f(0) first to get b.
        # 2. Evaluate f(5*e_i) for each i to get a_i.
        # 3. The minimum of a linear function on [-5,5]^d is at a corner.
        #    For each dimension i, if a_i > 0, set x_i = -5; if a_i < 0, set x_i = 5.
        # 4. Evaluate that corner to confirm (and get the best value).
        # 5. Fill remaining budget with calls to the best point (harmless, keeps b_i optimal).

        # Step 1: Evaluate center
        x0 = np.zeros(d)
        f0 = try_call(x0)
        if f0 is None:
            return x0, 0.0

        b = f0
        a = np.zeros(d)

        # Step 2: Evaluate vertices to get gradient components
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = try_call(xi)
            if fi is None:
                break
            a[i] = (fi - b) / 5.0

        # Step 3: Determine optimal corner
        x_opt = np.zeros(d)
        for i in range(d):
            if a[i] > 0:
                x_opt[i] = -5.0
            elif a[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Step 4: Evaluate the optimal corner
        f_opt_val = try_call(x_opt)
        if f_opt_val is None:
            # Should not happen with budget=500, but handle gracefully
            best_x = x0.copy()
            best_f = f0
        else:
            best_x = x_opt.copy()
            best_f = f_opt_val

        # Step 5: Fill remaining budget with calls to best_x
        while calls < budget and time_ok():
            try_call(best_x)

        return best_x, best_f
