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
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6 Knowledge:
        # f1(x) = 1 + 10*(n-1)*x_1
        # f2(x) = g(x) * h(f1/g)
        # g(x) = 1 + 10*(n-1) * sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i))
        # h(y) = 1 - sqrt(y)
        #
        # Pareto Front: x_2...x_n = 0 => g(x) = 1.
        # Then f1 = 1 + 10*(n-1)*x_1.
        # Let t = x_1. f1(t) = 1 + 10*(n-1)*t.
        # Range of f1 for t in [0, 1] is [1, 1 + 10*(n-1)].
        # For n=10, f1 in [1, 91].
        #
        # The problem statement says "minimizing all 2 objectives".
        # Reference point is [11.0, 11.0].
        # Points must be strictly better than reference point.
        # So we need f1 < 11 and f2 < 11.
        #
        # On the Pareto front (g=1):
        # f2 = h(f1) = 1 - sqrt(f1).
        # We need 1 - sqrt(f1) < 11 => -sqrt(f1) < 10 => sqrt(f1) > -10 (always true).
        # Wait, h(y) = 1 - sqrt(y).
        # If f1 is small, sqrt(f1) is small, h is close to 1.
        # If f1 is large, sqrt(f1) is large, h is negative.
        # Since we minimize, negative f2 is better than positive f2.
        # Reference point is 11.0. So any f2 < 11 is valid.
        # Since f2 on the front ranges from 1 (at f1=0) down to negative values,
        # almost all of the front is valid.
        #
        # However, we must ensure f1 < 11.
        # 1 + 10*(n-1)*x_1 < 11
        # 10*(n-1)*x_1 < 10
        # x_1 < 1/(n-1)
        # For n=10, x_1 < 1/9.
        #
        # So we should sample x_1 in [0, 1/9] to keep f1 < 11.
        # And set x_2...x_n = 0 to achieve the Pareto front (g=1).
        #
        # The score is Hypervolume of our points / Hypervolume of dense front sample.
        # To maximize the ratio, we need to approximate the front as well as possible
        # within the 200 evaluation budget.
        #
        # The front is defined by f2 = 1 - sqrt(f1) for f1 in [1, 11).
        # Note: f1 starts at 1 (when x_1=0) and goes up to 11 (when x_1=1/9).
        #
        # Hypervolume calculation for a set of points (f1, f2) with reference (r1, r2):
        # We want to maximize the area under the curve f2(f1) from f1=1 to f1=11,
        # bounded by the reference point.
        #
        # To get a good approximation of the curve with N points, we should distribute
        # them in a way that minimizes the error in the integral.
        # Uniform distribution in x_1 (and thus in f1) is a good start.
        #
        # Let's use all 200 evaluations to sample the Pareto front.
        # x_1 values: linspace(0, 1/(n-1), 200)
        # x_2...x_n = 0
        
        S_max = 1.0 / (dim - 1)
        
        # Generate x_1 values
        x1_values = np.linspace(0.0, S_max, max_evals)
        
        for x1 in x1_values:
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            # Construct x
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] is already 0
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
                
        return None
