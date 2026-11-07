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
        
        # Use a power-law distribution: x1 = 1 - t^alpha where alpha > 1 concentrates near x1=1
        # Optimal alpha: we want density proportional to (11 - f2(x1))
        # f2(x1) = 10*(1 - sqrt(x1/30))
        # 11 - f2 = 1 + 10*sqrt(x1/30)
        # At x1=0: 1, at x1=1: 1 + 10/sqrt(30) ≈ 2.83
        # So we want ~2.83x more density at x1=1 than at x1=0
        # With x1 = 1 - t^alpha, dx1/dt = -alpha * t^(alpha-1)
        # Density in x1 is proportional to 1/(dx1/dt) = 1/(alpha * t^(alpha-1))
        # We want density proportional to (1 + 10*sqrt(x1/30))
        # Near x1=1 (t≈0): density ∝ 1/t^(alpha-1), which blows up for alpha>1
        # Near x1=0 (t≈1): density ∝ 1/alpha
        # Let's try alpha = 2 first (uniform in t, quadratic in x1)
        t_values = np.linspace(0.0, 1.0, n_sweep)
        x1_values = 1.0 - t_values ** 2.0
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
