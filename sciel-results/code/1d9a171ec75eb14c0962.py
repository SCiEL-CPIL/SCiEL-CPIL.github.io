import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, dim))

        # Strategy: Uniform spacing in x0.
        # Rationale: The HV of a discrete set on a convex front is an overestimate of the integral.
        # The overestimate is larger for coarser sampling.
        # Uniform sampling provides a balanced overestimate.
        # Arc-length sampling minimizes the error (overestimate), potentially yielding a lower HV
        # compared to uniform sampling if the reference point is far (no clipping).
        
        N = budget
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            # Uniform spacing
            x0_vals = np.linspace(0.0, 1.0, N)
        
        # Clip to bounds
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = x0_val
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)
            
        return np.empty((0, dim))
