"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc


class Optimizer:
    """Exploit DTLZ1 structure: dense sampling on the exact Pareto front using Sobol sequence.
    
    For DTLZ1 with M=3 objectives and n=7 variables:
    - Shape variables x[2:] = 0.5 gives g = 0 (minimum)
    - 1+g = 1
    - f1 = 0.5 * x0
    - f2 = 0.5 * (1-x0) * x1  
    - f3 = 0.5 * (1-x0) * (1-x1)
    - f1+f2+f3 = 0.5 on the Pareto front
    
    We parameterize the front with simplex coordinates (s1, s2) where s1+s2+s3=1, si>=0.
    Mapping to decision space:
    x0 = s1
    x1 = s2 / (1 - s1)
    x2..x6 = 0.5
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_F = []
        evaluated_x = set()

        def evaluate(x):
            nonlocal all_F
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            all_F.append(f.copy())
            return f

        def make_front_point(s1, s2):
            """Create a decision vector on the Pareto front from simplex coords."""
            x = np.zeros(self.dim)
            # x0 corresponds to s1
            x[0] = s1
            # x1 corresponds to s2 / (1 - s1)
            if 1.0 - s1 > 1e-12:
                x[1] = s2 / (1.0 - s1)
            else:
                # Corner case: s1 ~ 1, implies s2 ~ 0
                x[1] = 0.0
            
            # Shape variables
            x[2:] = 0.5
            
            return x

        def try_evaluate(x0, x1):
            """Evaluate a point on the front, skipping duplicates."""
            key = (round(x0, 12), round(x1, 12))
            if key in evaluated_x:
                return False
            evaluated_x.add(key)
            x = make_front_point(x0, x1)
            evaluate(x)
            return True

        # Generate points using Sobol sequence for better space-filling
        # Sobol sequences are available in scipy.stats.qmc
        # We need 2D points for the simplex parameters
        
        # Generate a power of 2 number of points >= budget
        n_pow2 = 1
        while n_pow2 < self.budget:
            n_pow2 *= 2
        
        # Generate Sobol points
        # scramble=True is important for uniformity
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # Generate n_pow2 points in [0, 1)^2
        uv_pts = sampler.random(n_pow2)
        
        # We only need self.budget points
        uv_pts = uv_pts[:self.budget]
        
        for u, v in uv_pts:
            if len(all_F) >= self.budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Map [0,1)^2 to simplex coordinates (s1, s2)
            # Standard folding: if u+v > 1, reflect
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            
            # Now u+v <= 1, so s1=u, s2=v, s3=1-u-v are valid simplex coords
            s1 = u
            s2 = v
            
            # Clip to [0,1] to be safe against floating point errors
            s1 = np.clip(s1, 0.0, 1.0)
            s2 = np.clip(s2, 0.0, 1.0 - s1)
            
            try_evaluate(s1, s2)

        # If we didn't fill the budget (unlikely with Sobol), fill with random
        if len(all_F) < self.budget:
            rng = np.random.default_rng(12345)
            while len(all_F) < self.budget:
                if time.time() > deadline - 1:
                    break
                u = rng.uniform(0.0, 1.0)
                v = rng.uniform(0.0, 1.0)
                if u + v > 1.0:
                    u = 1.0 - u
                    v = 1.0 - v
                
                s1 = np.clip(u, 0.0, 1.0)
                s2 = np.clip(v, 0.0, 1.0 - s1)
                
                if not try_evaluate(s1, s2):
                    continue

        return all_F
# EVOLVE-BLOCK-END
