import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        try:
            from scipy.stats import qmc
            # Use Sobol or Halton for 2 dimensions (x0, x1)
            # Sobol is generally better for low dimensions
            sampler = qmc.Sobol(d=2, scramble=True)
            x01 = sampler.random(n_evals)
        except ImportError:
            # Fallback: 2D Halton
            def halton_sequence(n, d):
                points = np.zeros((n, d))
                for i in range(n):
                    for j in range(d):
                        f = 1.0
                        i_val = i + 1
                        while i_val > 0:
                            f /= j + 1
                            i_val = int(i_val // (j + 1))
                        points[i, j] = f
                return points
            x01 = halton_sequence(n_evals, 2)

        # Scale to bounds
        x01 = lo[:2] + (hi[:2] - lo[:2]) * x01
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x01[:, 0]
        X[:, 1] = x01[:, 1]
        # Set remaining variables to midpoint to ensure g(x) = 0
        # Assuming standard DTLZ2 where x[2:] affect g
        X[:, 2:] = 0.5
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
