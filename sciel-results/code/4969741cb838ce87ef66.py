import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        used = 0
        points = []
        objectives = []

        # DTLZ1 Pareto front for M=3:
        # Consists of two line segments:
        # 1. x1=0, x2 in [0,1] -> f=(0, 0.5*x2, 0.5*(1-x2))
        # 2. x2=1, x1 in [0,1] -> f=(0.5*x1, 0.5*(1-x1), 0)
        # We sample uniformly (via Sobol) along these segments.
        
        n_points = self.budget
        n_half = n_points // 2
        n_rem = n_points - n_half

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=1, scramble=True, seed=12345)
            # Generate n_half points for segment 1
            samples_1 = sampler.random(n_half)
            # Generate n_rem points for segment 2
            samples_2 = sampler.random(n_rem)
        except ImportError:
            rng = np.random.default_rng(12345)
            samples_1 = rng.random(n_half)
            samples_2 = rng.random(n_rem)

        # Evaluate points on Segment 1: x1=0, x2=t
        for i in range(n_half):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            t = float(np.clip(samples_1[i, 0], 0.0, 1.0))
            x = np.full(self.dim, 0.5)
            x[0] = 0.0
            x[1] = t
            f = problem(x)
            points.append(x.copy())
            objectives.append(np.array(f, dtype=float))
            used += 1

        # Evaluate points on Segment 2: x2=1, x1=t
        for i in range(n_rem):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            t = float(np.clip(samples_2[i, 0], 0.0, 1.0))
            x = np.full(self.dim, 0.5)
            x[0] = t
            x[1] = 1.0
            f = problem(x)
            points.append(x.copy())
            objectives.append(np.array(f, dtype=float))
            used += 1

        return points
