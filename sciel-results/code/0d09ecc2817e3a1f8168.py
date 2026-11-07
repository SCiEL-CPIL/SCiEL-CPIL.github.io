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
        # Use about 30% of budget for random sampling
        n_random = max(10, budget // 3)
        
        # Initial random samples
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)
        
        # Phase 2: Local optimization from best points found
        # Use L-BFGS-B from multiple starting points
        # Collect top starting points from random sampling
        # For simplicity, we'll just use the best point and a few nearby perturbations
        
        n_local = min(5, (budget - calls) // 50)  # Limit number of local optimizations
        local_starts = []
        
        # Add the best point
        local_starts.append(best_x.copy())
        
        # Add some perturbed versions of the best point
        for i in range(n_local - 1):
            if calls >= budget or time.time() > deadline:
                break
            perturbation = rng.uniform(-1, 1, dim) * 0.5
            x = np.clip(best_x + perturbation, -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            local_starts.append(x)
        
        # Run L-BFGS-B from each starting point
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
                    options={'maxiter': 100, 'maxfun': max(10, budget - calls - 10)}
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
