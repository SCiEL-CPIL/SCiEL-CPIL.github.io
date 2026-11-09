import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate diverse starting points with strategic ordering
        # Rosenbrock minimum is at (1,...,1) in standard form; shifted/rotated here.
        # With shift in [-4,4], true min is at (1+s,...,1+s) for some s.
        # So (1,...,1) is a very good starting point.
        
        rng = np.random.RandomState(42)
        
        # Priority-ordered starts:
        starts = []
        
        # 1. Standard Rosenbrock min (most likely if shift is small)
        starts.append(np.ones(self.dim))
        
        # 2. Small perturbations around (1,...,1) to handle small shifts
        for scale in [0.3, 0.7, 1.5]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        
        # 3. Origin (center of box)
        starts.append(np.zeros(self.dim))
        
        # 4. Random in likely region [-3, 5] (where min likely is)
        for _ in range(4):
            starts.append(rng.uniform(-3, 5, self.dim))
        
        # 5. Random in full box for robustness
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # 6. A few more structured points
        starts.append(np.ones(self.dim) * 0.5)
        starts.append(np.ones(self.dim) * 1.5)
        
        # Remove duplicates and clip to bounds
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5.0, 5.0)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Run L-BFGS-B from each starting point, prioritizing early starts
        for idx, x0 in enumerate(unique_starts):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            
            remaining_budget = self.budget - calls_made
            if remaining_budget < 15:
                break
            
            # Allocate budget: earlier starts get more budget
            # First 3 starts get up to 100 calls each, rest get 50
            if idx < 3:
                max_fun = min(100, remaining_budget)
            else:
                max_fun = min(50, remaining_budget)
            
            if max_fun < 10:
                break
            
            try:
                result = minimize(
                    safe_eval,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={
                        'maxiter': 200,
                        'maxfun': max_fun,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue
        
        # Final high-precision polish if we have budget and time
        if best_x is not None and calls_made < self.budget and time.time() - start_time < time_limit:
            remaining = self.budget - calls_made
            if remaining >= 20:
                try:
                    result = minimize(
                        safe_eval,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={
                            'maxiter': 500,
                            'maxfun': remaining,
                            'gtol': 1e-10,
                            'ftol': 1e-14,
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        # Ensure best_x is set (shouldn't happen, but safety)
        if best_x is None:
            best_x = np.ones(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
