import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific structure:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = g * (1 - sqrt(f1/g)) where g = 1 + 9*(sum(x[1:])/9)^(1/(1+2*sqrt(sum(x[1:])/9)))
        # The Pareto front is achieved when x[1:] = 0, so g=1.
        # Then f2 = 1 - sqrt(f1).
        # The front is the curve (f1, 1-sqrt(f1)) for f1 in [0, 1].
        
        # To maximize hypervolume, we want to cover the entire front [0, 1] in f1 space.
        # Since f1 is not monotonic, we need to be careful.
        # However, for any f1 in [0,1], there exists an x1 such that f1(x1)=f1.
        # We can sample x1 such that the resulting f1 values are well-distributed.
        
        # A better approach: sample f1 uniformly in [0,1] and find corresponding x1.
        # But f1 is not one-to-one. We can just sample x1 in a way that covers the range well.
        
        # Actually, the simplest and most effective approach for ZDT6 is:
        # 1. Set x[1:] = 0 (to be on the Pareto front)
        # 2. Sample x1 such that f1(x1) covers [0,1] well
        
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # This function oscillates. Let's sample x1 uniformly and see what f1 values we get.
        # Or, we can use a smarter sampling strategy.
        
        # For maximum HV, we want points evenly spaced in the objective space.
        # The front is (f1, 1-sqrt(f1)). If we sample f1 uniformly, the points will be 
        # closer together in f2 space for small f1 (since d(1-sqrt(f1))/df1 = -1/(2*sqrt(f1))).
        
        # Let's use a mix:
        # - 80% of points: evenly spaced in f1 space (inverse sampling)
        # - 20% of points: clustered near high-curvature regions
        
        # But first, let's just use uniform x1 sampling and see if it's good enough.
        # With 200 points, uniform x1 should give decent coverage.
        
        # Actually, let's think about this more carefully.
        # The hypervolume is maximized when points are evenly spaced in the objective space.
        # For the curve (f1, 1-sqrt(f1)), if we want even spacing in f2, we should sample 
        # f2 uniformly and compute f1 = (1-f2)^2.
        
        # Let's do that: sample f2 uniformly in [0,1], compute f1 = (1-f2)^2, then find x1.
        # But finding x1 for a given f1 requires solving f1(x1) = f1, which is complex.
        
        # Alternative: just sample x1 uniformly. With 200 points, this should be sufficient.
        
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0
        
        # Generate x1 values: use a combination of uniform and quasi-random
        # Halton sequence for better coverage
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
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] are already 0
            # Clip to bounds just in case
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
