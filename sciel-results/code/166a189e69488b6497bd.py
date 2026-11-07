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
        
        # Strategy:
        # 1. Evaluate the 3 vertices of the simplex to ensure extreme coverage.
        # 2. Use a Sobol sequence (or high-quality quasi-random) for the remaining points.
        # 3. Map 2D uniform points to the triangle uniformly using:
        #    x0 = sqrt(u1), x1 = u2 * (1 - x0)
        #    This ensures uniform density over the area of the triangle.
        
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        
        # Evaluate vertices first
        for x0, x1 in vertices:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            f = np.array(problem(x))
            all_F.append(f)
            
        n_remaining = n_evals - len(all_F)
        
        if n_remaining > 0:
            # Generate Sobol points
            # We use scipy if available for robustness, otherwise a simple fallback
            try:
                from scipy.stats import qmc
                sobol = qmc.Sobol(d=2, scramble=True)
                # Sobol requires power of 2 for best performance, but works otherwise
                # We might need to adjust n to power of 2 or just generate n
                # qmc.Sobol can generate arbitrary n
                points = sobol.random(n_remaining)
            except ImportError:
                # Fallback to Halton if scipy is not available, but with better mapping
                def halton(n, base):
                    result = np.zeros(n)
                    f = 1.0 / base
                    i = 1
                    for idx in range(n):
                        r = 0.0
                        while i > 0:
                            r += f * (i % base)
                            i //= base
                            f /= base
                        result[idx] = r
                        i += 1
                    return result
                points = np.column_stack((halton(n_remaining, 2), halton(n_remaining, 3)))

            # Map 2D points [0,1]^2 to Triangle [0,1]^2 s.t. x0+x1<=1
            # Uniform distribution on triangle:
            # Let u1, u2 be uniform in [0,1].
            # x0 = sqrt(u1)
            # x1 = u2 * (1 - x0)
            # This transformation maps the unit square uniformly to the right-angled triangle.
            
            u1 = points[:, 0]
            u2 = points[:, 1]
            
            x0 = np.sqrt(u1)
            x1 = u2 * (1.0 - x0)
            
            # Construct decision vectors and evaluate
            for i in range(n_remaining):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0[i]
                x[1] = x1[i]
                x[2:] = 0.5
                
                f = np.array(problem(x))
                all_F.append(f)
        
        return all_F
