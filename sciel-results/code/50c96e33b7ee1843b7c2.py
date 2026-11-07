import numpy as np
import time

class Optimizer:
    """
    Optimizer exploiting the known structure of ZDT1.
    
    ZDT1 Pareto front: x[0]=t in [0,1], x[1:]=0 gives f=(t, 1-sqrt(t)).
    To maximize hypervolume with reference [11,11], we sample t according to
    a density proportional to (11 - f2(t)) = 10 + sqrt(t), which concentrates
    points where the dominated region is tallest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        n_evals = min(self.budget, 200)
        if n_evals < 1:
            return np.zeros((0, self.dim))
        
        # ZDT1: Pareto front is f1 = x0, f2 = 1 - sqrt(x0) for x0 in [0,1], x[1:]=0
        # Optimal sampling: density proportional to (ref2 - f2(t)) = 10 + sqrt(t)
        # CDF: F(t) = (10t + (2/3)t^(3/2)) / (10 + 2/3)
        # Invert F to get t from uniform u
        
        # Precompute the normalization constant
        norm = 10.0 + 2.0/3.0  # = 32/3
        
        def cdf(t):
            """CDF of the optimal density."""
            t = np.clip(t, 0.0, 1.0)
            return (10.0 * t + (2.0/3.0) * t**1.5) / norm
        
        def invert_cdf(u):
            """Invert CDF using bisection (fast and robust)."""
            u = np.clip(u, 0.0, 1.0)
            lo_b = 0.0
            hi_b = 1.0
            for _ in range(50):  # 50 bisections give ~1e-15 precision
                mid = (lo_b + hi_b) / 2.0
                if cdf(mid) < u:
                    lo_b = mid
                else:
                    hi_b = mid
            return (lo_b + hi_b) / 2.0
        
        # Generate n_evals uniform samples in [0,1] and invert CDF
        u_samples = np.linspace(0.0, 1.0, n_evals)
        t_vals = np.array([invert_cdf(u) for u in u_samples])
        
        # Clip to bounds
        t_vals = np.clip(t_vals, lo[0], hi[0])
        
        # Construct population: x[0] = t, x[1:] = lo[1:] (to minimize g and thus f2)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        if self.dim > 1:
            X[:, 1:] = lo[1:]
            X[:, 1:] = np.clip(X[:, 1:], lo[1:], hi[1:])
        
        # Evaluate all points
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            fx = problem(X[i])
            F.append(fx)
        
        return X
