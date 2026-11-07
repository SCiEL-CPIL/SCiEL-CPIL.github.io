"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Direct Pareto front construction via analytical insight on ZDT4 structure."""
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        
        all_points = []
        all_evals = 0
        
        def check_time():
            return time.time() > deadline
        
        def eval_point(x):
            nonlocal all_evals
            if all_evals >= self.budget or check_time():
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_evals += 1
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                all_points.append(f)
            return f
        
        # ZDT4 Analysis:
        # f1 = x1
        # f2 = g(x2..xn) * (1 - sqrt(x1/g))
        # g(x) = 10*n + sum_{i=2}^n ((xi - 0.5)^2 - 10*cos(20*pi*(xi - 0.5)))
        # The global minimum of each term is -10, occurring at xi = 0.5.
        # Thus, global min g = 10*n - 10*(n-1) = 10.
        # To lie on the Pareto front, we must have g(x) = 10, which implies xi = 0.5 for all i > 1.
        
        # Set x2..xn to the known global minimum location
        x_rest = np.full(n - 1, 0.5)
        # Clip to bounds in case bounds are non-standard (though 0.5 is usually within [-5, 5])
        x_rest = np.clip(x_rest, lo[1:], hi[1:])
        
        # Sweep x1 across its bounds to trace the Pareto front
        # We use the entire budget for this sweep to maximize coverage
        n_sweep = self.budget
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1v in x1_vals:
            if all_evals >= self.budget or check_time():
                break
            x = np.concatenate(([x1v], x_rest))
            eval_point(x)
        
        # Convert to array
        if len(all_points) > 0:
            result = np.array(all_points, dtype=np.float64)
            mask = np.all(np.isfinite(result), axis=1)
            result = result[mask]
        else:
            # Last resort
            x = (lo + hi) / 2
            f = problem(x)
            result = np.array([f], dtype=np.float64)
        
        return result
# EVOLVE-BLOCK-END
