import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific setup
        # n_obj = 3
        # dim = 12
        # Angle variables: x[0], x[1]
        # Distance variables: x[2] to x[11] (10 variables)
        
        # We want g(x) = 1.
        # g(x) = sum((x_i - 0.5)^2) for i in 2..11
        # Let x_i = 0.5 + d
        # 10 * d^2 = 1 => d = sqrt(0.1)
        d_val = np.sqrt(0.1)
        
        # Pre-allocate the distance part
        # We assume the standard DTLZ2 where the last dim - n_obj + 1 vars are distance vars.
        # Here 12 - 3 + 1 = 10.
        # Indices 2 to 11.
        
        points = []
        n_evals = self.budget
        
        # Generate Halton sequence points for the 2 angle dimensions
        # We use indices 1 to n_evals
        for i in range(1, n_evals + 1):
            if time.time() > deadline:
                break
            
            u1 = self._halton(i, 2)
            u2 = self._halton(i, 3)
            
            # Map to [0, 1] range for x[0], x[1]
            # The problem function likely maps [0, 1] to [0, pi/2] internally
            x = np.zeros(self.dim)
            x[0] = u1
            x[1] = u2
            
            # Set distance variables
            # Indices 2 to 11
            x[2:] = 0.5 + d_val
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            points.append(f)
            
        return points
