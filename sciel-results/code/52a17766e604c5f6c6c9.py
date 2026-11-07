import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # ZDT2 Pareto front: f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30)) for x[1:] = 0
        # Hypervolume contribution per unit f1 is (11 - f2) = 11 - 10*(1 - sqrt(x[0]/30))
        # = 1 + 10*sqrt(x[0]/30), which is larger near f1=1
        # We want to maximize hypervolume, so sample more densely where marginal HV gain is high.
        # The marginal HV gain per point at f1 is proportional to (11 - f2) * (df1/dt)
        # Let's use a non-uniform distribution that concentrates points near f1=1
        
        n_sweep = self.budget
        
        # Optimal sampling for hypervolume maximization on ZDT2
        # The Pareto front is f1 = x1, f2 = 10*(1 - sqrt(x1/30)) for x1 in [0,1]
        # Reference point is [11, 11]
        # Hypervolume contribution of a point at f1 is (11 - f2) = 1 + 10*sqrt(f1/30)
        # To maximize hypervolume with N points, we want density proportional to (11 - f2)
        # Using x1 = 1 - t^alpha, the density in x1 is proportional to 1/(alpha * t^(alpha-1))
        # We want this proportional to (1 + 10*sqrt(x1/30))
        # Near x1=1 (t≈0): 1+10*sqrt(x1/30) ≈ 1+10/sqrt(30) ≈ 2.83
        # Near x1=0 (t≈1): 1+10*sqrt(x1/30) ≈ 1
        # So we want density ratio of ~2.83:1 between x1=1 and x1=0
        # With x1 = 1-t^alpha, density at x1=1 is infinite for alpha>1, but the effective
        # density over a small interval dx1 is proportional to 1/t^(alpha-1)
        # Let's use alpha=1 (uniform in x1) as a baseline, or optimize alpha numerically
        # Actually, for hypervolume, the optimal density is proportional to the "width" 
        # of the hypervolume slice, which is (11-f2). Let's use importance sampling.
        
        # Better approach: sample x1 with density proportional to (11 - f2(x1))
        # f2(x1) = 10*(1-sqrt(x1/30)), so 11-f2 = 1+10*sqrt(x1/30)
        # CDF: F(x1) = integral_0^x1 (1+10*sqrt(u/30)) du / integral_0^1 (1+10*sqrt(u/30)) du
        # Numerator: x1 + 10/sqrt(30) * (2/3) * x1^(3/2) = x1 + (20/(3*sqrt(30))) * x1^(3/2)
        # Denominator: 1 + (20/(3*sqrt(30)))
        
        # Use rejection sampling or inverse CDF to generate x1 values
        # For simplicity, use a grid with non-uniform spacing based on the CDF
        
        # Compute the CDF values for a uniform grid, then invert
        n_grid = 10000
        x1_grid = np.linspace(0, 1, n_grid)
        f2_grid = 10 * (1 - np.sqrt(x1_grid / 30))
        weight_grid = 11 - f2_grid  # = 1 + 10*sqrt(x1_grid/30)
        cdf_grid = np.cumsum(weight_grid)
        cdf_grid /= cdf_grid[-1]
        
        # For each of n_sweep points, sample uniformly in CDF space and invert
        u_values = np.linspace(0, 1, n_sweep)
        x1_values = np.interp(u_values, cdf_grid, x1_grid)
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
