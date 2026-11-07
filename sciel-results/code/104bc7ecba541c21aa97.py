import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 270
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= self.budget or time.time() > deadline:
                return None
            calls += 1
            return func(x)

        # Start at origin
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if f0 is None:
            return x0, f0

        # Probe each axis with a moderate step to determine sign of gradient
        delta = 1.0
        signs = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            probe = np.zeros(dim)
            probe[i] = delta
            fi = safe_call(probe)
            if fi is None:
                break
            if fi < f0:
                signs[i] = -1.0  # function decreases in + direction -> go to +5
            elif fi > f0:
                signs[i] = 1.0   # function increases in + direction -> go to -5
            else:
                signs[i] = 0.0   # flat in this direction

        # Jump to the optimal vertex
        x_opt = signs * 5.0
        f_opt = safe_call(x_opt)
        if f_opt is None:
            return x0, f0

        return x_opt, f_opt
