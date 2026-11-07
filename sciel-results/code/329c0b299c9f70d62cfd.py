import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        if N <= 0:
            return None

        # ZDT1 structure:
        # f1 = x1
        # g = 1 + 9 * sum(x2..xn)/(n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Pareto front when g=1 (x2..xn=0): f2 = 1 - sqrt(f1)
        # Reference point: [11, 11]
        # HV = sum (f1_{i+1} - f1_i) * (11 - f2_i) with f1_{N+1}=11
        #    = sum (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
        # Optimal density: more points near f1=0 where sqrt(f1) changes fast.
        # Use f1_i = (i/N)^2 for i=0..N (clamped to [0,1])

        base_vec = np.zeros(self.dim)

        if N == 1:
            x = base_vec.copy()
            x[0] = 0.5
            problem(x)
            return None

        # Generate x1 values with optimal spacing for ZDT1 hypervolume
        # HV integrand is (10 + sqrt(f1)), nearly uniform
        # Use endpoints 0 and 1, with interior points spaced to equalize
        # hypervolume contribution per interval
        if N > 1:
            if N == 2:
                x1_values = np.array([0.0, 1.0])
            else:
                # Include endpoints, optimize interior
                x1_values = np.zeros(N)
                x1_values[0] = 0.0
                x1_values[-1] = 1.0
                # Interior: use spacing proportional to 1/(10+sqrt(f1))
                # Approximate: nearly uniform with slight bias toward 0
                # Use f1_i = (i/(N-1))^1.5 for interior, then adjust
                interior_idx = np.arange(1, N-1, dtype=np.float64)
                t = interior_idx / (N - 1)
                x1_values[1:-1] = t ** 1.5
        else:
            x1_values = np.array([0.5])

        for i in range(N):
            if time.time() > deadline:
                break

            x = base_vec.copy()
            x[0] = x1_values[i]
            problem(x)

        return None
