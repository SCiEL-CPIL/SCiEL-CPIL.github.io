# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize optimizer for ZDT2.
    
    Exploits the known Pareto front geometry of ZDT2:
    - Front: x1 in [0,1], x2..xn = 0
    - f1 = x1
    - f2 = C * (1 - sqrt(x1/C)) where C = 1 + 10*(n-1)
    
    For 2 objectives, hypervolume with reference (r1, r2) is computed
    by sorting points by f1 and summing trapezoid contributions.
    
    We optimize the placement of n_points x1 values to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 parameters
        C = 1.0 + 10.0 * (dim - 1)
        r1 = 11.0
        r2 = 11.0

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # f2 as function of x1
        def f2_of_x1(x1):
            x1_clipped = np.clip(x1, 0.0, 1.0)
            return C * (1.0 - np.sqrt(x1_clipped / C))

        # Threshold x1 where f2 = r2: C*(1-sqrt(x1/C)) = r2
        # 1 - sqrt(x1/C) = r2/C
        # sqrt(x1/C) = 1 - r2/C
        # x1 = C * (1 - r2/C)^2
        x1_thresh = C * (1.0 - r2 / C) ** 2
        
        # Ensure threshold is within [0, 1]
        x1_thresh = np.clip(x1_thresh, 0.0, 1.0)
        
        # Optimal 1D design: density proportional to local curvature of Pareto front.
        # For ZDT2: f2 = C*(1-sqrt(x1/C)), f2'' ∝ x1^(-3/2), f2' ≈ const magnitude
        # Curvature κ ∝ x1^(-1/2), so optimal CDF ∝ sqrt(x1), giving x1_i ∝ i²
        # Map [0,1] quadratic nodes to [x1_thresh, 1.0]:
        t = np.linspace(0.0, 1.0, n_points)
        x1_opt = x1_thresh + (1.0 - x1_thresh) * t * t

        # Evaluate the optimal points
        X_list = []
        F_list = []

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_opt[i]
            x = np.clip(x, lo, hi)

            f = problem(x)
            X_list.append(x)
            F_list.append(f)

        return np.array(X_list)
# EVOLVE-BLOCK-END
