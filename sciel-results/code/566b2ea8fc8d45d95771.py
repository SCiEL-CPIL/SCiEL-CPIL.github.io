import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        rng = np.random.RandomState(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                return 1e10
            call_count += 1
            if call_count > self.budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Multi-start exploration
        # Use ~70% of budget for exploration with multiple random starts
        explore_budget = int(0.7 * self.budget)
        num_starts = 8
        calls_per_start = max(30, explore_budget // num_starts)
        
        for s in range(num_starts):
            if call_count >= explore_budget or time.time() > deadline:
                break
            
            # Random start point in [-4, 4]^d
            x0 = rng.uniform(-4, 4, size=self.dim)
            
            # Adaptive initial simplex size based on remaining budget
            delta = 1.5 if s < num_starts // 2 else 0.8
            
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Limit iterations to stay within per-start budget
            max_iter = max(10, calls_per_start // (self.dim + 2))
            
            try:
                minimize(wrapped_func, x0, method='Nelder-Mead',
                         options={'initial_simplex': simplex, 'maxiter': max_iter,
                                  'xatol': 1e-3, 'fatol': 1e-3})
            except Exception:
                pass
            
            # If we've used too many calls, stop early
            if call_count >= explore_budget:
                break
        
        # Phase 2: Final high-precision polish from best point found
        if call_count < self.budget and time.time() < deadline:
            polish_budget = self.budget - call_count
            if polish_budget > 20:
                # Start from best_x with tighter simplex
                x0 = best_x.copy()
                delta = 0.3
                simplex = [x0.copy()]
                for i in range(self.dim):
                    xi = x0.copy()
                    xi[i] += delta
                    simplex.append(xi)
                
                max_iter = max(20, polish_budget // (self.dim + 2))
                
                try:
                    minimize(wrapped_func, x0, method='Nelder-Mead',
                             options={'initial_simplex': simplex, 'maxiter': max_iter,
                                      'xatol': 1e-6, 'fatol': 1e-6})
                except Exception:
                    pass
        
        return best_x, best_f
