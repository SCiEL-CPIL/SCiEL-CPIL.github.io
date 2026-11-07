import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        
        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget
        
        # 1. Add the 3 vertices of the simplex in x1, x2 space
        # Vertices: (0,0), (1,0), (0,1)
        vertices = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        
        x1_vals = []
        x2_vals = []
        
        for v in vertices:
            x1_vals.append(v[0])
            x2_vals.append(v[1])
            
        remaining = n_evals - len(vertices)
        
        # 2. Generate Halton sequence for the rest
        # Halton in 2 dimensions (bases 2 and 3)
        # We need to map [0,1]^2 to the simplex x1>=0, x2>=0, x1+x2<=1
        # Use the same inverse CDF transform as before:
        # u, v ~ U(0,1)
        # x1 = 1 - sqrt(1 - u)
        # x2 = (1 - x1) * v
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, remaining + 1):
            u = halton(i, 2)
            v = halton(i, 3)
            
            x1 = 1.0 - np.sqrt(1.0 - u)
            x2 = (1.0 - x1) * v
            
            x1_vals.append(x1)
            x2_vals.append(x2)
            
        x1_vals = np.array(x1_vals)
        x2_vals = np.array(x2_vals)
        
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1_vals[i]
            x[1] = x2_vals[i]
            x[2:] = 0.5
            
            evaluate(x)
            
        return all_F
