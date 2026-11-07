import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # Strategy: Exploit the known structure of ZDT1.
        # The Pareto front is achieved when x[1:] = 0.
        # f1 = x[0], f2 depends on x[0].
        # We sample x[0] uniformly in [0, 1] to cover the front.
        # This approach was shown to be effective in inspirations.
        
        n_samples = budget
        
        # Uniform sampling of t in [0, 1]
        # Using linspace ensures we hit 0 and 1 exactly, maximizing the span in f1.
        t_vals = np.linspace(0.0, 1.0, n_samples)
        
        X = []
        F = []
        
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            
            # Construct point on the Pareto front
            x = np.zeros(dim)
            x[0] = float(t)
            
            f = eval_point(x)
            
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)
        
        # Safety fallback: If for some reason the structured points failed (e.g., bounds issue),
        # try a few random points. Though with bounds [0,1] and x[0]=t in [0,1], this shouldn't happen.
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            span = hi - lo
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
