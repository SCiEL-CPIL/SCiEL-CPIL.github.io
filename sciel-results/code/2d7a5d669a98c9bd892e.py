import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        # Evaluator kills after 330s. We should aim for ~300s max.
        self.time_limit = 280.0

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.clip(x, -5.0, 5.0).copy()

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Structural Change: Pure Multi-start L-BFGS-B
        # 1. Generate a diverse set of initial points.
        # 2. For each point, run L-BFGS-B.
        # 3. If a better solution is found, use it as the next start point (adaptive restart).
        
        # Initial diverse starts:
        # - Origin
        # - Random points
        # - Grid points (corners and midpoints) to ensure coverage
        starts = []
        starts.append(np.zeros(n))
        
        # Add random points
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.uniform(-5.0, 5.0, size=n)
            starts.append(x)
            
        # Add some grid-like points to catch different basins
        # Simple 3x3... grid is too expensive, so pick specific corners and edges
        for i in range(n):
            for j in range(n):
                if i != j:
                    x = np.zeros(n)
                    x[i] = 4.0
                    x[j] = 4.0
                    starts.append(x)
                    x[i] = -4.0
                    x[j] = -4.0
                    starts.append(x)
        
        # Deduplicate starts
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Shuffle to mix deterministic and random
        np.random.shuffle(unique_starts)
        
        # Allocation:
        # We have ~500 calls.
        # L-BFGS-B typically uses 10-50 calls to converge if close.
        # We can afford ~10-20 restarts.
        # Let's cap the number of L-BFGS-B runs to avoid running out of budget on one.
        max_runs = len(unique_starts)
        
        for i in range(max_runs):
            if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                break
            
            # Select start point
            if i < len(unique_starts):
                x0 = unique_starts[i]
            else:
                # If we have more runs than starts, restart from best
                x0 = self.best_x.copy()
                # Add small noise to escape local minima if stuck
                x0 += np.random.normal(0, 0.1, size=n)
                x0 = np.clip(x0, -5.0, 5.0)

            # Determine budget for this L-BFGS-B run
            # Leave some buffer for potential final polish
            calls_remaining = budget - self.call_count
            # Cap individual run to prevent one long run from consuming all budget
            # but allow it to go deep if it's the first few runs
            maxfun = min(calls_remaining, 50 + i * 5) 
            
            if maxfun < 5:
                break

            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxfun': maxfun, 'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                    
            except Exception:
                pass

        # Final Polish: Use remaining budget on the absolute best
        if self.call_count < budget - 5 and self.best_x is not None and time.time() - self.start_time < self.time_limit:
            remaining_final = budget - self.call_count
            if remaining_final > 10:
                try:
                    # Tighter tolerance for final polish
                    res = minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxfun': remaining_final, 'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
