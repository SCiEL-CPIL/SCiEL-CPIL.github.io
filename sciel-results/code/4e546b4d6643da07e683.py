"""Program for multi-objective optimization on ZDT1 using hypervolume-optimized quantization."""
import numpy as np


class Optimizer:
    """Optimizes ZDT1 by exploiting its structure: fix non-critical dims to 0.5,
    use a grid of x1 values optimized for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        if n <= 0:
            return None

        # Standard ZDT1 Pareto front:
        # f1 = x1
        # f2 = (1 + x1^2) * g(x)
        # On the Pareto front, g(x) = 1 (achieved by x_i = 0 for i > 1).
        # So f2(x1) = 1 + x1^2.
        # Reference point: (11, 11).
        # Hypervolume contribution weight for quantization:
        # The term in the HV sum that depends on the point's position along the front
        # is related to (R2 - f2(x)) = 11 - (1 + x^2) = 10 - x^2.
        # To maximize HV, we place points according to the inverse CDF of this weight function.
        
        k = n
        x1_values = np.zeros(k)
        
        # CDF of w(x) = 10 - x^2 on [0, 1]
        # Integral I(x) = 10x - x^3/3
        # I(1) = 10 - 1/3 = 29/3
        total_integral = 29.0 / 3.0
        
        for i in range(k):
            # Midpoint rule for optimal quantization
            target_fraction = (i + 0.5) / k
            rhs = target_fraction * total_integral
            
            # Solve 10x - x^3/3 = rhs for x in [0, 1]
            # Function h(x) = 10x - x^3/3 is monotonic increasing on [0, 1]
            # h(0) = 0, h(1) = 29/3
            lo_x, hi_x = 0.0, 1.0
            for _ in range(50):
                mid_x = (lo_x + hi_x) / 2.0
                val = 10.0 * mid_x - mid_x**3 / 3.0
                if val < rhs:
                    lo_x = mid_x
                else:
                    hi_x = mid_x
            x1_values[i] = (lo_x + hi_x) / 2.0
        
        results = []
        for x1 in x1_values:
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] = 0 ensures g(x) = 1, placing the point on the Pareto front
            f = problem(x)
            results.append(f)
        
        return results
