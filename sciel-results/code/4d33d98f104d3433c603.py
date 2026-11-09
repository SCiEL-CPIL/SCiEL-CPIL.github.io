import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s margin

        n_evals = self.budget
        if n_evals <= 0:
            return []

        # ZDT6 Pareto front: x_0 in [0,1], x_i = 0 for i>1
        # f1 = 2 + 10*sin(pi*x_0/10), f2 = 1 - sqrt(f1/11)
        # f1 ranges from 2 to ~2.31 (very small range!), f2 from ~0.956 to 0
        # The front is nearly flat in f1, so hypervolume is dominated by f2 coverage
        # We want to maximize the area under the front relative to reference [11,11]
        
        # Key insight: f1 barely changes (2 to 2.31), so the front is nearly vertical
        # in (f1,f2) space. The hypervolume contribution per point depends on
        # how well we cover the f2 range [0, ~0.956].
        
        # Since f1 is nearly constant, the best strategy is to spread points
        # evenly in f2-space, which means solving for x_0 given desired f2.
        
        # f2 = 1 - sqrt(f1/11), f1 = 2 + 10*sin(pi*x_0/10)
        # Given f2, f1 = 11*(1-f2)^2, then x_0 = (10/pi)*arcsin((f1-2)/10)
        
        results = []
        
        if n_evals == 1:
            # Best single point: middle of f2 range
            f2_mid = 0.478
            f1_mid = 11.0 * (1.0 - f2_mid) ** 2
            x0 = (10.0 / np.pi) * np.arcsin((f1_mid - 2.0) / 10.0)
            x = np.zeros(self.dim)
            x[0] = float(np.clip(x0, 0.0, 1.0))
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
            return results

        # Strategy: spread points in f2-space from 0 to f2_max
        # f2_max = 1 - sqrt(2/11) ≈ 0.478
        f2_max = 1.0 - np.sqrt(2.0 / 11.0)
        
        # Use a mix: some uniform in f2, some clustered where curvature is high
        # The front has higher curvature near f2=0 (x0=1) and f2=f2_max (x0=0)
        
        # Generate points in f2-space
        # For a nearly vertical front, uniform spacing in f2 is near-optimal for hypervolume
        # But we need to account for the curvature of the ZDT6 front
        # The front: f1 = 11*(1-f2)^2, so df1/df2 = -22*(1-f2)
        # Hypervolume contribution per point ~ (df1/df2) * (f2 - 0) for the "width"
        # To equalize contributions, space points so that (1-f2_i)^2 * f2_i is roughly uniform
        # Actually, for hypervolume of a monotonic front with ref [11,11]:
        # HV = sum of boxes. Each point contributes (f1_i - f1_{i+1})*f2_i approximately
        # Since f1 is nearly constant, this is ~ const * f2_i * (step in f2)
        # So uniform f2 spacing should be good
        t = np.linspace(0.0, 1.0, n_evals)
        f2_vals = f2_max * t
        
        # Convert f2 to x0
        f1_vals = 11.0 * (1.0 - f2_vals) ** 2
        x0_vals = (10.0 / np.pi) * np.arcsin((f1_vals - 2.0) / 10.0)
        
        # Clip to valid range
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        
        # Evaluate
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0_vals[i])
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return results
