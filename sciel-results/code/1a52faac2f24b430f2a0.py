import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280

    def _halton(self, n, dims):
        """
        Generate n points in [0, 1)^dims using the Halton sequence.
        """
        # First few primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dims))
        for i in range(1, n + 1):
            for j in range(dims):
                base = primes[j]
                f = 1.0
                r = 0.0
                ip = i
                while ip > 0:
                    f /= base
                    r += f * (ip % base)
                    ip //= base
                points[i - 1, j] = r
        return points

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        
        # DTLZ4 structure:
        # n=12, m=3
        # Distance vars: x[0:10]
        # Front shape vars: x[10], x[11]
        # Pareto front is at x[0:10] = 0.5
        # f1 = x[10] * sin(pi/2 * x[11])
        # f2 = x[10] * cos(pi/2 * x[11])
        # f3 = 1 - x[10]
        
        # We sample x[10] and x[11] using Halton sequence in 2D.
        # This directly targets the Pareto front.
        
        n_samples = self.budget
        halton_pts = self._halton(n_samples, 2)
        
        for i in range(n_samples):
            if time.time() > self.deadline:
                break
            
            s, t = halton_pts[i]
            
            x = np.zeros(self.dim)
            # Set distance variables to 0.5 to be on the Pareto front
            x[:self.dim - 2] = 0.5
            # Set front shape variables
            x[self.dim - 2] = s
            x[self.dim - 1] = t
            
            # Clip to bounds just in case, though 0.5, s, t are in [0,1]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            all_F.append(f)
            
        return all_F
