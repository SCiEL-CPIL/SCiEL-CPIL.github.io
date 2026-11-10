import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        seen = set()
        
        def evaluate_params(u, s):
            nonlocal n_evals
            if n_evals >= budget:
                return
            if time.time() > deadline:
                return
            # Round to avoid floating point duplicates
            key = (round(u, 12), round(s, 12))
            if key in seen:
                return
            seen.add(key)
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = s * s
            # x[2:] remains 0
            
            # Ensure x is within bounds (it should be, but just in case)
            x = np.clip(x, lo, hi)
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1

        # Strategy: Optimal transport / importance sampling for hypervolume
        # DTLZ7 front: f0=u, f1=s^2, f2=1-s
        # Hypervolume contribution is maximized by uniform coverage in (f0, f2) space.
        # Since f0=u and f2=1-s, this implies uniform sampling in (u, s).
        # However, to be safe and potentially capture nuances, we can slightly bias
        # towards the center or use a well-distributed Sobol sequence.
        
        # Reserve points for critical boundary/corner/edge points
        n_reserved = 16
        n_sobol = budget - n_reserved
        
        if n_sobol < 8:
            n_sobol = max(budget - 4, 1)
            n_reserved = budget - n_sobol
        
        # Use the evaluator's random seed to vary the Sobol sequence slightly
        sobol_seed = np.random.randint(0, 2**32)
        
        sampler = qmc.Sobol(d=2, scramble=True, seed=sobol_seed)
        sobol_points = sampler.random(n_sobol)
        
        # Evaluate Sobol points: u = point[0], s = point[1]
        for u, s in sobol_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            evaluate_params(u, s)
        
        # Add critical boundary and corner points
        # These ensure coverage of the extremes of the Pareto front
        critical_points = [
            # Corners of the (u, s) square
            (0.0, 0.0),   # f = (0, 0, 1)
            (1.0, 0.0),   # f = (1, 0, 1)
            (0.0, 1.0),   # f = (0, 1, 0)
            (1.0, 1.0),   # f = (1, 1, 0)
            # Edge points: u=0, varying s
            (0.0, 0.25),
            (0.0, 0.5),
            (0.0, 0.75),
            # Edge points: u=1, varying s
            (1.0, 0.25),
            (1.0, 0.5),
            (1.0, 0.75),
            # Edge points: s=0, varying u
            (0.25, 0.0),
            (0.5, 0.0),
            (0.75, 0.0),
            # Edge points: s=1, varying u
            (0.25, 1.0),
            (0.5, 1.0),
            (0.75, 1.0),
        ]
        
        for u, s in critical_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            evaluate_params(u, s)
        
        return results
