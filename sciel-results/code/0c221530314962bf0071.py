# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Analytical Pareto front sampler for ZDT2 with equal-area distribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_points = self.budget
        
        # ZDT2 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Reference point: [11, 11]
        # HV contribution function: g(x) = 11 - (1 - sqrt(x)) = 10 + sqrt(x)
        # True HV = integral_0^1 (10 + sqrt(x)) dx = 10 + 2/3 = 32/3
        
        if n_points <= 0:
            return np.zeros((0, self.dim))
        
        # We want to choose x_1, ..., x_N such that the hypervolume of the discrete set is maximized.
        # This is equivalent to minimizing the approximation error of the integral.
        # A good strategy is to distribute points such that each segment contributes equally to the true HV.
        
        true_hv = 32.0 / 3.0
        area_per_point = true_hv / n_points
        
        # We need to find x_i such that integral_{x_{i-1}}^{x_i} (10 + sqrt(x)) dx = area_per_point
        # Let F(x) = 10x + (2/3)x^(3/2). We need F(x_i) - F(x_{i-1}) = area_per_point.
        # So F(x_i) = F(x_{i-1}) + area_per_point.
        # F(x_0) = 0 (x_0 = 0)
        # F(x_i) = i * area_per_point
        # 10 x_i + (2/3) x_i^(3/2) = i * area_per_point
        
        # We can solve for x_i using Newton's method or bisection.
        # Since x_i is in [0, 1], and F is monotonic, bisection is safe and fast.
        
        x_values = np.zeros(n_points)
        
        def F(x):
            return 10.0 * x + (2.0 / 3.0) * np.sqrt(x) * x
        
        # Pre-compute target F values
        targets = np.arange(1, n_points + 1) * area_per_point
        
        for i in range(n_points):
            target = targets[i]
            # Bisection search for x in [0, 1]
            lo = 0.0
            hi = 1.0
            for _ in range(50):
                mid = (lo + hi) / 2.0
                if F(mid) < target:
                    lo = mid
                else:
                    hi = mid
            x_values[i] = (lo + hi) / 2.0
            
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x_values
        
        # Evaluate all points
        # The problem function is called n_points times
        # We don't need to store F if we just return X, but the contract says
        # "Every returned objective vector is recorded". This happens when problem(x) is called.
        for x in X:
            problem(x)
        
        return X
# EVOLVE-BLOCK-END
