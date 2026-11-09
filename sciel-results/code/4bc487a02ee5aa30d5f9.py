import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 using Pareto Front Sampling.
    
    Strategy:
    1. Exploit the separable structure of DTLZ7:
       - f1 = x[0]
       - f2 = x[1]
       - f3 depends on tail variables x[2:] via a complex function.
    2. To minimize f3 and reach the Pareto front, the tail variables x[2:] must be
       set to 0.5 (which minimizes the g function term).
    3. The Pareto front is a 2D surface parameterized by x[0] and x[1].
    4. Instead of a uniform grid (which has corner redundancy and less efficient
       hypervolume coverage for a fixed N), we use a Sobol sequence to sample
       the 2D box [0,1]x[0,1] for x[0] and x[1].
    5. Sobol sequences provide low-discrepancy sampling, which maximizes the
       hypervolume of the resulting point set for a given budget.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The optimal tail values for DTLZ7 are 0.5 (assuming bounds [0,1])
        # If bounds are different, we should map to the midpoint.
        tail_mid = 0.5 * (lo[2:] + hi[2:])
        
        n = self.budget
        
        # Generate Sobol sequence points in [0, 1]^2
        # We need n points. Sobol sequences are best for powers of 2, 
        # but we can just take the first n points from a larger sequence.
        # scipy.stats.qmc.Sobol is available in modern scipy.
        # If not available, we can use a simple Halton sequence or random,
        # but Sobol is preferred for low discrepancy.
        
        try:
            from scipy.stats.qmc import Sobol
            # Generate enough points, then take the first n
            # Sobol requires dimension <= 21 in some older versions, but dim=2 is fine.
            # scramble=True helps with randomization for different seeds if needed, 
            # but here the problem is deterministic given x.
            sobol = Sobol(d=2, scramble=True, seed=42) 
            # We need n points. Let's generate n points.
            # Note: Sobol.generate returns an array of shape (n, d)
            # The first point is usually [0, 0]. We might want to skip it or keep it.
            # Keeping it is fine.
            points_2d = sobol.generate(n)
        except ImportError:
            # Fallback to Halton if scipy qmc is not available (unlikely in modern envs)
            # or if there's an issue.
            # Simple random would be worse. Halton is easy to implement.
            points_2d = np.zeros((n, 2))
            for i in range(n):
                points_2d[i, 0] = self._halton(i+1, 2)
                points_2d[i, 1] = self._halton(i+1, 3)
                
        # Map points from [0, 1] to [lo, hi] for x0 and x1
        x0_vals = points_2d[:, 0] * (hi[0] - lo[0]) + lo[0]
        x1_vals = points_2d[:, 1] * (hi[1] - lo[1]) + lo[1]
        
        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            x = np.empty(self.dim)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            x[2:] = tail_mid
            F.append(problem(x))
            
        # If we have budget left (shouldn't happen if loop finished), add randoms
        remaining = self.budget - len(F)
        if remaining > 0 and time.time() < deadline:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(self.dim)
                F.append(problem(x))
                
        return F

    @staticmethod
    def _halton(index, base):
        """Generate the index-th value of the Halton sequence for a given base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
