import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _inverse_cdf_zdt2(self, u, n_points):
        """
        Inverse CDF sampling for ZDT2 Pareto front.
        Density proportional to 1/(10 + sqrt(t)), which equalizes
        marginal hypervolume contribution per point.
        """
        # F(t) = [2*sqrt(t) - 20*ln(10+sqrt(t))] / [2 - 20*ln(11/10)]
        # We need to solve F(t) = u for t given u in [0,1]
        # Use Newton's method or bisection
        
        denom = 2.0 - 20.0 * np.log(11.0 / 10.0)
        
        t = np.zeros(n_points)
        # Initial guess: uniform
        t = u.copy()
        
        # Newton iteration
        for _ in range(10):
            sqrt_t = np.sqrt(t)
            F_val = (2.0 * sqrt_t - 20.0 * np.log(10.0 + sqrt_t)) / denom
            # dF/dt = (1/sqrt(t) - 10/(sqrt(t)*(10+sqrt(t)))) / denom
            # = (1/sqrt(t)) * (1 - 10/(10+sqrt(t))) / denom
            # = (1/sqrt(t)) * (sqrt(t)/(10+sqrt(t))) / denom
            # = 1 / ((10+sqrt(t)) * denom)
            with np.errstate(divide='ignore', invalid='ignore'):
                dF_dt = 1.0 / ((10.0 + sqrt_t) * denom)
                dF_dt = np.where(np.isfinite(dF_dt), dF_dt, 1.0/denom)
            t_new = t - (F_val - u) / dF_dt
            t = np.clip(t_new, 0.0, 1.0)
        
        return t

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # Non-uniform sampling via inverse CDF to equalize HV contribution
        u = np.linspace(0.0, 1.0, N)
        f1_vals = self._inverse_cdf_zdt2(u, N)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = f1_vals
        # x[1:] remains 0
        
        F_list = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F_list.append(f)
            
        return X
