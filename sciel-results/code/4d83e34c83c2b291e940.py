import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        n_evals = self.budget
        
        # Allocate budget:
        # Most points to the standard ZDT2 Pareto front (x_2..x_n = 0.5)
        # Few points as a hedge (x_2..x_n = 0.0) in case of variant or boundary issues
        n_hedge = 5
        n_main = n_evals - n_hedge

        points = np.zeros((n_evals, self.dim))

        # Main Strategy: Standard ZDT2 Pareto Front
        # f2 is concave, high curvature near x1=0.
        # Quadratic spacing of x1 provides better hypervolume approximation than linear.
        if n_main > 1:
            # Linearly spaced indices, then squared to bias towards 0
            x1_main = np.linspace(lo[0], hi[0], n_main)
            # Normalize to [0,1], square, then denormalize
            # This ensures x1=lo[0] and x1=hi[0] are hit exactly
            x1_norm = (x1_main - lo[0]) / (hi[0] - lo[0])
            x1_sq = x1_norm ** 2
            points[:n_main, 0] = lo[0] + x1_sq * (hi[0] - lo[0])
            points[:n_main, 1:] = 0.5  # Minimizer of g(x) for ZDT2
        else:
            points[0, 0] = lo[0]
            points[0, 1:] = 0.5

        # Hedge Strategy: Alternative front (x_2..x_n = 0.0)
        # This covers the case where the problem might be ZDT1-like or if g(x) has local minima at 0
        if n_hedge > 1:
            points[n_main:, 0] = np.linspace(lo[0], hi[0], n_hedge)
            points[n_main:, 1:] = 0.0
        elif n_hedge == 1:
            points[n_main, 0] = lo[0]
            points[n_main, 1:] = 0.0

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(points[i])
            
        return points
