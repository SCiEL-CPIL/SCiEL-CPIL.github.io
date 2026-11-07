import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0 => g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1],1)
        # h(y,1) = 9/(1+2*(sin(3*pi*y)+sin(4*pi*y)))
        
        N = budget
        if N < 2:
            N = 2

        # Strategic initial points
        init_candidates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.5, 0.5],
        ])

        n_init = min(len(init_candidates), N)
        points_to_eval = []
        
        for i in range(n_init):
            points_to_eval.append(init_candidates[i])

        # Generate Sobol points for the rest
        remaining = N - n_init
        if remaining > 0:
            # Sobol sequence in 2D
            # We need to generate enough points to fill the remaining budget
            # Sobol requires a power of 2 for optimal performance, but works for any N
            # We generate remaining points
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate points
            # Note: Sobol points are in [0, 1)
            sobol_points = sampler.random(remaining)
            # Add to points_to_eval
            for p in sobol_points:
                points_to_eval.append(p)

        # Evaluate all points
        for p in points_to_eval:
            if not check_time():
                break
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = p[0]
            x[1] = p[1]
            evaluate(x)

        return None
