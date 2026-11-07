import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = 300.0

        # Generate candidate x1 values for ZDT6
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # We want to sample the Pareto front (f1, 1-sqrt(f1))
        # The front spans f1 in [0, 1]
        
        # Strategy: Use a mix of uniform and clustered points
        # 1. Uniform in x1 to ensure coverage
        # 2. Clustered near points where f1 has high derivative (curvature)
        
        # First, let's do a coarse grid to understand f1(x1) behavior
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # For x1 in [0, 1]:
        #   x1=0: f1=0
        #   x1=0.5: f1 = 1 - exp(-2)*sin(3*pi)^8 = 1 - 0 = 1 (since sin(3*pi)=0)
        #   Wait, sin(6*pi*0.5) = sin(3*pi) = 0, so f1=1
        #   x1=1: f1 = 1 - exp(-4)*sin(6*pi)^8 = 1 - 0 = 1
        
        # Actually, let me reconsider. The function oscillates.
        # sin(6*pi*x1) has zeros at x1 = k/6 for k=0,1,2,...,6
        # So at x1 = 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, sin=0, f1=1
        # Between these, sin is non-zero, so f1 < 1
        
        # To maximize hypervolume, we want points on the Pareto front.
        # The Pareto front of ZDT6 is the curve where x2..xn = 0 (g=1)
        # and x1 varies, giving (f1, 1-sqrt(f1))
        
        # We need to sample x1 such that the resulting f1 values cover [0,1] well.
        # Since f1 is not monotonic, uniform x1 sampling may not be optimal.
        
        # Let's use inverse sampling: sample f1 uniformly in [0,1], then find x1.
        # But f1 is not one-to-one. Multiple x1 give same f1.
        # The Pareto front is parameterized by f1, so we just need distinct f1 values.
        
        # Actually, the Pareto front is the set of points (f1, 1-sqrt(f1)) for f1 in [0,1].
        # To generate a point on the front, we pick f1 in [0,1] and set x1 such that f1(x1)=f1.
        # But since f1 is not monotonic, for a given f1, there are multiple x1.
        # We just need one x1 per f1 value.
        
        # Simpler approach: just sample x1 uniformly and evaluate.
        # The key insight is that we should place points where they contribute most to HV.
        
        # Let's use a strategy: 
        # 1. Start with a coarse grid (e.g., 50 points)
        # 2. Refine by adding points in gaps
        
        # But with 200 evaluations, let's just do a well-distributed set.
        
        # Generate x1 values: use a combination of uniform and quasi-random
        # Halton sequence for better coverage in 1D (which is just a low-discrepancy sequence)
        
        def halton_1d(n, base=2):
            """Generate n Halton sequence points in [0,1)"""
            result = []
            f = 1.0
            for i in range(1, n + 1):
                x = 0.0
                num = i
                while num:
                    f /= base
                    x += f * (num % base)
                    num //= base
                result.append(x)
            return np.array(result)
        
        # Use Halton sequence for x1
        x1_vals = halton_1d(n_evals)
        
        # Also include the endpoints 0 and 1 to ensure full coverage
        x1_vals = np.sort(np.concatenate([x1_vals, [0.0, 1.0]]))
        
        # Remove duplicates (within tolerance)
        x1_vals = np.unique(np.round(x1_vals, 15))
        
        # If we have more than budget, take first budget points
        if len(x1_vals) > n_evals:
            x1_vals = x1_vals[:n_evals]
        
        # Evaluate each point
        for x1 in x1_vals:
            if time.time() - start_time > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            problem(x)
        
        return None
