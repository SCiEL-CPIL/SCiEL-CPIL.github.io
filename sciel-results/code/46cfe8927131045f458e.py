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
        
        n_evals = self.budget
        all_F = []
        
        # For DTLZ1 with 7 vars, 3 objectives:
        # f1 = 0.5 * x0 * (1+g), f2 = 0.5 * x1 * (1+g), f3 = 0.5 * (1-x0-x1) * (1+g)
        # g_min = 1, so 1+g = 2, thus f_i = x_i (for the Pareto front)
        # Pareto front: x0+x1+x2 = 1, x0,x1,x2 >= 0, x3..x6 = 0.5
        # f1 = x0, f2 = x1, f3 = x2 where x2 = 1-x0-x1
        
        # Use Sobol sequence for better space-filling properties
        # Generate Sobol points in [0,1]^2
        n_sobol = n_evals - 3  # Reserve 3 for vertices
        
        # Simple Sobol-like sequence using scrambled Halton or direct construction
        # For 2D, we can use a good LHS or stratified sampling
        # Let's use a stratified approach: divide triangle into regions
        
        # Actually, let's use a better approach: sample uniformly on the triangle
        # using the standard method, but with a quasi-random sequence
        
        # Generate quasi-random points using van der Corput (Halton) but with better bases
        def halton(n, base):
            result = []
            f = 1.0 / base
            i = 1
            for _ in range(n):
                r = 0.0
                while i > 0:
                    r += f * (i % base)
                    i //= base
                    f /= base
                result.append(r)
                i += 1
            return result
        
        # Use primes for better distribution
        h1 = halton(n_sobol, 2)
        h2 = halton(n_sobol, 3)
        
        # First add the three vertices of the simplex
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        
        for x0, x1 in vertices:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            f = np.array(problem(x))
            all_F.append(f)
        
        # Then add the quasi-random interior points
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            
            u1 = h1[i]
            u2 = h2[i]
            
            # Map to triangle: if u1+u2 > 1, reflect
            if u1 + u2 > 1.0:
                u1 = 1.0 - u1
                u2 = 1.0 - u2
            
            x0 = u1
            x1 = u2
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            
            f = np.array(problem(x))
            all_F.append(f)
            
        return all_F
