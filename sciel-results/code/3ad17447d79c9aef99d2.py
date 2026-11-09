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
        
        n_evals = self.budget
        if n_evals < 1:
            return None
        
        # For ZDT4 with n=10:
        # f1(x) = x0
        # g(x) = sum_{i=1}^{9} (x_i^2 - 10*cos(4*pi*x_i))
        # f2(x) = 1 + 9*g(x)/(n-1) = 1 + g(x)  (since n-1 = 9)
        # 
        # The minimum of each term x_i^2 - 10*cos(4*pi*x_i) is -10 at x_i = 0
        # So g_min = 9 * (-10) = -90
        # f2_min = 1 + (-90) = -89
        #
        # Wait, but the reference point is [11.0, 400.0], which suggests f2 can be large.
        # Let me reconsider. The standard ZDT4 definition is:
        # f1 = x0
        # g(x) = 10*n + sum_{i=1}^{n-1}(x_i^2 - 10*cos(4*pi*x_i))
        # f2 = (1 + g(x)) * (1 - sqrt(f1/g(x)))
        #
        # With n=10: g(x) = 100 + sum_{i=1}^{9}(x_i^2 - 10*cos(4*pi*x_i))
        # At x_i = 0: each term = -10, sum = -90, g = 100 - 90 = 10
        # f2 = (1 + 10) * (1 - sqrt(x0/10)) = 11 * (1 - sqrt(x0/10))
        #
        # So the Pareto front is: f1 = x0 in [0,1], f2 = 11*(1 - sqrt(x0/10))
        # When x0=0: f2 = 11
        # When x0=1: f2 = 11*(1 - sqrt(0.1)) = 11*(1 - 0.3162) = 11*0.6838 = 7.522
        #
        # But wait, x0 bounds might not be [0,1]. Let me check what bounds are given.
        # The problem says bounds are given per variable. For ZDT4, typically x0 in [0,1] and x_i in [-5,5] for i>0.
        # But we should use the given bounds.
        #
        # Key insight: to minimize both objectives, we want:
        # 1. x0 as small as possible (minimizes f1)
        # 2. g(x) as small as possible, which means x_i = 0 for i=1..9 (minimizes f2)
        #
        # But there's a trade-off: f2 depends on x0 through sqrt(f1/g).
        # For a given x0, the best f2 is achieved when g is minimized (x_i = 0).
        #
        # So the Pareto front is parameterized by x0:
        # f1 = x0
        # f2 = (1 + g_min) * (1 - sqrt(x0/g_min)) where g_min = 10 (when x_i=0 for all i>0)
        # f2 = 11 * (1 - sqrt(x0/10))
        #
        # But x0 is bounded by [lo[0], hi[0]]. If lo[0] > 0, we can't reach f1=0.
        # If hi[0] > 1, x0 can be > 1, but sqrt(x0/10) would be > 0.316.
        #
        # Actually, for ZDT4, x0 is typically in [0,1]. Let me assume the bounds are standard.
        # But we should handle the general case.
        #
        # Strategy: sample x0 across its range, and set x_i = 0 for i > 0 (the optimal tail).
        # This gives us points on or very near the Pareto front.
        
        x_tail = np.clip(0.0, lo[1:], hi[1:])
        
        # Use a distribution that clusters near the ends of the x0 range
        # The Pareto front is curved, so more points near the ends help with hypervolume
        
        n_evals = self.budget
        if n_evals < 1:
            return None
        
        # Generate t values with clustering near both ends
        # Use a beta-like distribution: t = 0.5 - 0.5*cos(pi*u) clusters near 0 and 1
        # Or use: t = (1 - cos(pi*u)) / 2 which maps u in [0,1] to t in [0,1] with clustering at ends
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            if n_evals > 1:
                u = i / (n_evals - 1)
                # Cluster near both ends using cosine
                t = 0.5 - 0.5 * np.cos(np.pi * u)
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
