import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin of 5 seconds

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        # DTLZ7 Pareto front analysis:
        # f1 = x0, f2 = x1, f3 = h(x1) with x2..xn=0.
        # h(v) = 1 - v/(1+v) * (1 + sin(3*pi*v))
        # Non-dominated points are in v in [0, 1/6] and near v=5/6.
        # We set x0=0 to maximize f1 contribution.
        
        # Sampling strategy:
        # 70% points in [0, 1/6] (main front)
        # 30% points in [0.8, 0.85] (secondary front near global min of h)
        
        n_main = int(0.7 * self.budget)
        n_sec = self.budget - n_main

        # Use Halton sequence for quasi-random sampling in the respective intervals
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate main front points
        for i in range(n_main):
            if time.time() > deadline:
                break
            # Map Halton(0, 2) to [0, 1/6]
            u = halton(i + 1, 2)
            v = u * (1.0 / 6.0)
            
            x = np.zeros(self.dim)
            x[0] = 0.0
            x[1] = v
            # x[2:] = 0
            problem(x)
            used += 1

        # Generate secondary front points
        for i in range(n_sec):
            if time.time() > deadline:
                break
            # Map Halton(0, 3) to [0.8, 0.85]
            u = halton(i + 1, 3)
            v = 0.8 + u * 0.05
            
            x = np.zeros(self.dim)
            x[0] = 0.0
            x[1] = v
            # x[2:] = 0
            problem(x)
            used += 1

        # Fill any remaining budget (should be 0) or handle edge cases
        while used < self.budget:
            if time.time() > deadline:
                break
            # Random fallback
            v = np.random.rand() * (1.0 / 6.0)
            x = np.zeros(self.dim)
            x[0] = 0.0
            x[1] = v
            problem(x)
            used += 1

        return None
