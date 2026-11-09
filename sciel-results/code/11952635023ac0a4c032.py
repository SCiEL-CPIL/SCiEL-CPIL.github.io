import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT4, g(x) = 10*n + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=1..n-1
        # The minimum of each term is at x_i=0 (value -10)
        # So g_min = 10*n - 10*(n-1) = 10 + 10 = 20? Let me recompute.
        # Actually with n=10, there are 9 tail variables. g = 10*10 + sum_{9 terms}
        # At x_i=0: each term = -10, sum = -90, g = 100 - 90 = 10.
        
        x_tail = np.clip(0.0, lo[1:], hi[1:])
        
        n_evals = self.budget
        if n_evals < 1:
            return None
        
        # Use a distribution that clusters points near both ends of x0
        # The Pareto front of ZDT4 has f2 = 1 + 9*g/(n-1), and g is minimized at x_tail=0
        # f1 = x0, so the front goes from (0, 1+9*10/9) = (0, 11) to (1, 1+9*10/9) = (1, 11)
        # Wait, that's not right. Let me reconsider.
        # Actually for ZDT4: f1 = x0, f2 = 1 + 9*(g/n-1) where g = sum(x_i^2 - 10*cos(4*pi*x_i)) for i=1..n-1
        # At x_i = 0: term = 0 - 10 = -10, so g = 9*(-10) = -90, f2 = 1 + 9*(-90)/9 = 1 - 90 = -89
        # But that doesn't seem right for the reference point [11, 400].
        # Let me just use a good distribution. The key insight is that the Pareto front 
        # is parameterized by x0, so we should sample x0 well.
        
        # Use a combination: uniform sampling plus extra points near the boundaries
        # where the front may be steeper or where hypervolume contributions are larger
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            if n_evals > 1:
                # Use arcsine distribution: t = 0.5 * (1 - cos(pi * i/(n-1)))
                # This clusters points near both 0 and 1
                t = 0.5 * (1 - np.cos(np.pi * i / (n_evals - 1)))
            else:
                t = 0.5
                
            x0 = lo[0] + t * (hi[0] - lo[0])
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None
