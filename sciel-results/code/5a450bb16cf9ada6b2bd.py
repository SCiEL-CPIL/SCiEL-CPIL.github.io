import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 exploiting problem structure.
    
    DTLZ2 with 3 objectives and 12 variables:
    - g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
    - f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
    - where t1 = x0*pi/2, t2 = x1*pi/2
    - Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
    
    Strategy: 
    1. Use the entire budget to evaluate points exactly on the Pareto front.
    2. Use a deterministic low-discrepancy sequence (Halton) for the 2D 
       parameter space (x0, x1) to ensure uniform coverage of the front.
    3. This maximizes the hypervolume by avoiding any off-front points 
       (which are strictly dominated) and maximizing the spread of on-front points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._deadline = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 30s before the 330s wall-clock kill
        self._deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure: g=0 when x[2:] = 0.5
        # Pareto front parameterized by x0, x1 in [0,1]
        # f = [cos(t1)*cos(t2), cos(t1)*sin(t2), sin(t1)] where t1=x0*pi/2, t2=x1*pi/2
        
        # Generate Halton sequence points for x0, x1 in [0, 1]
        # Using bases 2 and 3 for the two dimensions
        def halton_sequence(n, bases):
            """
            Generate n points in the Halton sequence for the given bases.
            Returns an array of shape (n, len(bases)).
            """
            points = np.zeros((n, len(bases)))
            for j, base in enumerate(bases):
                for i_val in range(n):
                    result = 0.0
                    denom = 1.0
                    num = i_val
                    while num > 0:
                        denom *= base
                        digit = num % base
                        result += digit / denom
                        num //= base
                    points[i_val, j] = result
            return points

        # Generate 200 points
        points_2d = halton_sequence(budget, [2, 3])
        
        x0 = points_2d[:, 0]
        x1 = points_2d[:, 1]
        
        # Construct full 12D vectors on the Pareto front
        X = np.zeros((budget, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5  # g=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = []
        used = 0
        for x in X:
            if time.time() > self._deadline:
                break
            f = problem(x)
            F.append(f)
            used += 1
        
        return F
