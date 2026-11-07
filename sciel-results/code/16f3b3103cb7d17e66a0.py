import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid multi-start optimizer: random sampling + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x) != dim:
                x = x[:dim]
            fx = func(x)
            calls += 1
            return float(fx)
        
        # Phase 1: Random sampling to find good starting points
        # Reduced random sampling to save budget for more local restarts
        n_random = 20
        
        random_points = []
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            random_points.append((fx, x))
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)
            random_points.append((best_f, best_x))

        # Sort random points by fitness to get best starting points for local optimization
        random_points.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization from best points found
        # Increase number of local restarts to explore more basins
        # Each L-BFGS-B run is limited to ~25 function calls
        max_fun_per_local = 25
        n_local = min(10, (budget - calls) // max_fun_per_local)
        
        # Use the top n_local points from random sampling
        local_starts = [p[1] for p in random_points[:n_local]]
        
        bounds = [(-5, 5)] * dim
        
        for start in local_starts:
            if calls >= budget or time.time() > deadline:
                break
            
            try:
                result = minimize(
                    safe_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': max(10, min(25, budget - calls - 5))}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final refinement if budget remains
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': max(10, budget - calls - 5)}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
