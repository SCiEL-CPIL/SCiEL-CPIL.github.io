import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        
        calls_made = 0
        best_x = None
        best_fx = float('inf')
        
        def safe_func(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            # Clip to bounds just in case, though func should handle it
            x = np.clip(x, -5, 5)
            fx = func(x)
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Initial deterministic samples
        # The unshifted minimum is at 2. The shift is hidden but minimizer is in [-4, 4]^5.
        # Try points around 2 and the center of the box.
        init_points = [
            np.zeros(self.dim),
            np.full(self.dim, 2.0),
            np.full(self.dim, -2.0),
            np.full(self.dim, 4.0),
            np.full(self.dim, -4.0),
        ]
        
        # Add a few random points
        rng = np.random.default_rng(42)
        for _ in range(5):
            init_points.append(rng.uniform(-4, 4, self.dim))

        for x0 in init_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            try:
                fx = safe_func(x0)
                update_best(x0, fx)
            except (TimeoutError, ValueError):
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = safe_func(best_x)

        # Phase 2: Local optimization from best point and other promising points
        # Use Nelder-Mead for local refinement
        # We will try to optimize from the top 3 best initial points
        # But we only kept the single best. Let's keep a list of candidates.
        
        # Let's restart the logic slightly to keep candidates
        candidates = []
        for x0 in init_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            try:
                fx = safe_func(x0)
                candidates.append((fx, x0.copy()))
            except (TimeoutError, ValueError):
                break
        
        # Sort candidates by fx
        candidates.sort(key=lambda x: x[0])
        
        # Take the best 3 candidates for local optimization
        for fx0, x0 in candidates[:3]:
            if time.time() > deadline or calls_made >= self.budget:
                break
            try:
                res = minimize(safe_func, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < best_fx:
                    update_best(res.x, res.fun)
            except (TimeoutError, ValueError):
                break

        # Phase 3: If budget remains, do more random restarts
        if calls_made < self.budget and time.time() < deadline:
            for _ in range(10): # Up to 10 more restarts
                if time.time() > deadline or calls_made >= self.budget:
                    break
                x0 = rng.uniform(-4, 4, self.dim)
                try:
                    res = minimize(safe_func, x0, method='Nelder-Mead', options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res.success or res.fun < best_fx:
                        update_best(res.x, res.fun)
                except (TimeoutError, ValueError):
                    break

        return best_x, best_fx
