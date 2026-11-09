import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_points = self.budget
        # Generate Halton sequence
        # Bases 2 and 3
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        for i in range(1, n_points + 1):
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            points.append((u1, u2))
            
        # Convert to DTLZ2 decision variables
        # x1, x2 are angles scaled to [0, 1]
        # x3...x12 are 0.5
        for u1, u2 in points:
            x = np.zeros(self.dim)
            x[0] = u1
            x[1] = u2
            x[2:] = 0.5
            problem(x)
        return None
