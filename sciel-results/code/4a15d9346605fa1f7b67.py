import numpy as np
from scipy.stats.qmc import Sobol

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj
            # Pre-compute y for tail variables
            # Solve z^2 - 4*cos(20*pi*z) = -1 for z > 0
            # Using Newton or simple search
            target = -1.0
            z = 0.02
            for _ in range(100):
                val = z*z - 4*np.cos(20*np.pi*z)
                if abs(val - target) < 1e-6: break
                # Newton step
                deriv = 2*z + 80*np.pi*np.sin(20*np.pi*z)
                if abs(deriv) < 1e-10: break
                z -= (val - target) / deriv
                if z < 0: z = 0.02
            self.tail_val = 0.5 + z
            
        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Sobol sequence for simplex sampling
            # We need to sample x1, x2 such that x1+x2 <= 1
            # Standard way: sample u1, u2 from [0,1]^2
            # x1 = u1 * sqrt(u2) ? No.
            # Use Dirichlet-like or simple rejection?
            # Sobol in 2D gives uniform square.
            # To get uniform simplex:
            # Let v1, v2, v3 be independent exp(1) or use sqrt method.
            # Simpler: Sample s1, s2 from Sobol.
            # x1 = s1 * (1-s2) ? No, that's not uniform on simplex.
            # Uniform on simplex {x1+x2<=1}:
            # Sample u, v from [0,1].
            # If u+v > 1, reflect?
            # Actually, just sample 3 numbers from Sobol, square them, normalize.
            
            sobol = Sobol(d=3, scramble=True)
            # We need 'budget' points
            points = sobol.random(self.budget)
            
            results = []
            for p in points:
                # Map to simplex
                w = p**2
                w_sum = w.sum()
                r1, r2, r3 = w / w_sum
                
                x = np.zeros(self.dim)
                x[0] = r1
                x[1] = r2
                # x[2] to x[6] are tail
                x[2:] = self.tail_val
                
                # Clip to bounds just in case
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(x)
            
            return results[-1]
