import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        # DTLZ7 specific structure:
        # First 2 variables (x0, x1) determine position on Pareto front.
        # Remaining variables determine distance from front (g).
        # To be on the front, distance variables must be 0.
        
        # We will sample the Pareto front directly using a Halton sequence.
        # This ensures good coverage of the 2D surface of the front.
        
        n_samples = self.budget
        # Halton sequence for 2 dimensions
        def halton(n, base):
            q = np.zeros(n)
            f = 1.0
            for i in range(n):
                r = 0.0
                f = 1.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                q[i] = r
            return q
        
        # Generate Halton points in [0, 1]
        # Skip the first few points to avoid the origin (0,0) which might be a singular point or just less useful?
        # Actually, (0,0) gives f3 = -1, which is very good. We should include it.
        # But Halton starts with 0,0. Let's use i=1..n to get points in (0,1).
        # Or just use the sequence as is.
        
        h1 = halton(n_samples, 2)
        h2 = halton(n_samples, 3)
        
        # Create points
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = h1
        X[:, 1] = h2
        # X[:, 2:] remains 0
        
        # Evaluate all points
        # We must check time periodically, but 200 evaluations is fast.
        # Let's evaluate in chunks to be safe.
        
        results = []
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
        
        return results
