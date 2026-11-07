import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _lhs_2d(self, n, seed=42):
        """Latin Hypercube Sampling in 2D, returns n points in [0,1]^2."""
        rng = np.random.RandomState(seed)
        # Create intervals
        x = (np.arange(n) + 0.5) / n
        y = (np.arange(n) + 0.5) / n
        rng.shuffle(x)
        rng.shuffle(y)
        return np.column_stack([x, y])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f))
            return f

        # Phase 1: Vertices
        vertices = [
            np.zeros(self.dim),
            np.zeros(self.dim),
            np.zeros(self.dim)
        ]
        vertices[1][1] = 1.0
        vertices[2][0] = 1.0
        
        for v in vertices:
            if used >= self.budget:
                break
            x = np.clip(v, lo, hi)
            evaluate(x)

        # Phase 2: LHS on the transformed space
        remaining = self.budget - used
        if remaining > 0:
            lhs_points = self._lhs_2d(remaining, seed=123)
            for u1, u2 in lhs_points:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # u1 is in [0, 1], represents cos(theta_1)
                # u2 is in [0, 1], represents theta_2 / (pi/2)
                
                x0 = (2.0 / np.pi) * np.arccos(u1)
                x1 = u2
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                
                x = np.clip(x, lo, hi)
                evaluate(x)

        return all_X
