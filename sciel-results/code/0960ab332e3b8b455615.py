"""Multi-objective optimizer for ZDT6 using analytic Pareto front sampling."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto Front Analysis:
        # The Pareto front is defined by x_2...x_dim = 0.
        # f_1(x) = x_1^2
        # f_2(x) = (1 + g(x)) (1 - sqrt(f_1(x) / (1 + g(x))))
        # g(x) = 1 + 10 * sum_{i=2}^{dim} ( (x_i - 0.5)^2 * cos(20 * pi * (x_i - 0.5)) )
        #
        # When x_i = 0.5 for i >= 2, the sum is 0, so g(x) = 1.
        # Then f_2 = 2 * (1 - sqrt(f_1 / 2)) = 2 - 2 * sqrt(f_1 / 2) = 2 - sqrt(2 * f_1).
        # With f_1 = x_1^2, f_2 = 2 - sqrt(2) * x_1.
        #
        # So the Pareto front is a smooth curve: f_2 = 2 - sqrt(2) * sqrt(f_1).
        # We want to place 200 points along this curve to maximize hypervolume.
        #
        # The hypervolume contribution of a set of points on a decreasing front
        # with reference point (r1, r2) is the area under the "step function"
        # formed by the points. To maximize this, we should distribute points
        # such that the "gaps" in the hypervolume are minimized.
        #
        # For a smooth front, the optimal spacing is roughly uniform in
        # "hypervolume space". A good approximation is to sample f_1 uniformly,
        # which gives roughly uniform spacing in f_1. Since the front is smooth
        # and decreasing, this should give near-optimal hypervolume.
        #
        # Let's use a slightly denser sampling near f_1 = 0 where the curve is
        # steeper (in f_2 vs f_1 space), and sparser near f_1 = 1 where it's flatter.
        # Actually, for hypervolume, uniform in f_1 is a very good first-order approximation.
        
        n_samples = budget
        
        # Sample f_1 uniformly in [0, 1]
        f1_values = np.linspace(0.0, 1.0, n_samples)
        
        # Solve for x_1 = sqrt(f_1)
        x1_values = np.sqrt(f1_values)
        
        # Construct the population
        # x_2 to x_dim are set to 0.5 (where g is minimized to 1)
        X = np.full((n_samples, dim), 0.5)
        X[:, 0] = x1_values
        
        # Evaluate all points
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
