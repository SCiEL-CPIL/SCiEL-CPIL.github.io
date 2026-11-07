import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 250.0  # Safety margin before 300s limit

    def __call__(self, func):
        d = self.dim
        calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Call 1: origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls += 1
        update_best(x0, f0)

        if time.time() > self.deadline or calls >= self.budget:
            return best_x, best_f

        # Determine sign of w_i for each dimension using boundary probes
        signs = np.zeros(d)  # +1 means x*_i = -5, -1 means x*_i = +5

        for i in range(d):
            if calls >= self.budget or time.time() > self.deadline:
                break

            # Probe positive boundary
            x_pos = np.zeros(d)
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            calls += 1
            update_best(x_pos, f_pos)

            if calls >= self.budget or time.time() > self.deadline:
                break

            # Probe negative boundary
            x_neg = np.zeros(d)
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            calls += 1
            update_best(x_neg, f_neg)

            # Determine sign of w_i
            if f_pos > f_neg:
                signs[i] = 1  # w_i > 0, so x*_i = -5
            else:
                signs[i] = -1  # w_i <= 0, so x*_i = +5

            # Evaluate partial vertex: first i+1 dims at optimal sign, rest at 0
            if i < d - 1:
                x_partial = np.zeros(d)
                for j in range(i + 1):
                    x_partial[j] = -5.0 if signs[j] == 1 else 5.0
                f_partial = func(x_partial)
                calls += 1
                update_best(x_partial, f_partial)

                if calls >= self.budget or time.time() > self.deadline:
                    break

        # Evaluate full optimal vertex
        if calls < self.budget and time.time() < self.deadline:
            x_opt = np.zeros(d)
            for i in range(d):
                x_opt[i] = -5.0 if signs[i] == 1 else 5.0
            f_opt = func(x_opt)
            calls += 1
            update_best(x_opt, f_opt)

        return best_x, best_f
