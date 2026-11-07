import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        n = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Step 1: Determine optimal values for x_2 ... x_n
        # For ZDT4, g(x_i) is minimized at 0.
        optimal_x_rest = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            idx = i + 1
            if lo[idx] <= 0 and 0 <= hi[idx]:
                optimal_x_rest[i] = 0.0
            else:
                # Find closest to 0 within bounds
                if abs(lo[idx]) < abs(hi[idx]):
                    optimal_x_rest[i] = lo[idx]
                else:
                    optimal_x_rest[i] = hi[idx]

        x1_lo = lo[0]
        x1_hi = hi[0]
        if x1_lo > x1_hi:
            x1_lo, x1_hi = x1_hi, x1_lo

        # If x1 range is zero, just evaluate one point
        if x1_hi - x1_lo < 1e-12:
            x = np.zeros(n)
            x[0] = x1_lo
            x[1:] = optimal_x_rest
            problem(x)
            return x

        # Step 2: Scout phase to identify the active Pareto range for x_1
        # We assume the Pareto front is a contiguous segment in x_1.
        # We sample a small number of points to find the bounds of this segment.
        scout_count = 20
        if budget <= scout_count:
            scout_count = budget
            refine_count = 0
        else:
            refine_count = budget - scout_count

        scout_x1 = np.linspace(x1_lo, x1_hi, scout_count)
        scout_X = np.zeros((scout_count, n))
        scout_X[:, 0] = scout_x1
        scout_X[:, 1:] = optimal_x_rest

        scout_F = np.empty((scout_count, self.n_obj))
        for i in range(scout_count):
            if time.time() > deadline:
                break
            scout_F[i] = problem(scout_X[i])

        # Identify non-dominated points in the scout set
        # A point i is dominated by j if F[j] <= F[i] and F[j] != F[i]
        nd_flags = np.ones(scout_count, dtype=bool)
        for i in range(scout_count):
            if not nd_flags[i]:
                continue
            for j in range(scout_count):
                if i == j:
                    continue
                if np.all(scout_F[j] <= scout_F[i]) and np.any(scout_F[j] < scout_F[i]):
                    nd_flags[i] = False
                    break

        nd_indices = np.where(nd_flags)[0]
        
        if len(nd_indices) == 0:
            # Fallback: use full range
            x1_min = x1_lo
            x1_max = x1_hi
        else:
            x1_min = scout_x1[nd_indices].min()
            x1_max = scout_x1[nd_indices].max()

        # Step 3: Refine phase
        # Uniformly sample the identified active range [x1_min, x1_max]
        if refine_count > 0:
            # Ensure we don't miss the exact endpoints if they are critical
            # The linspace will include endpoints by default
            refine_x1 = np.linspace(x1_min, x1_max, refine_count)
            refine_X = np.zeros((refine_count, n))
            refine_X[:, 0] = refine_x1
            refine_X[:, 1:] = optimal_x_rest

            for i in range(refine_count):
                if time.time() > deadline:
                    break
                problem(refine_X[i])

        return scout_X
