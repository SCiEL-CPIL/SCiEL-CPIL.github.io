import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(12345)

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial sampling to find a good starting point
        # Use Halton/Sobol-like deterministic sampling for good coverage
        n_init = 50
        # Use a simple deterministic grid mixed with random
        # First, try a coarse grid
        for i in range(n_init):
            # Mix of structured and random
            if i < 10:
                # Structured points
                x = np.zeros(n)
                # Try different scales and directions
                x = np.random.uniform(-5, 5, n)
            else:
                x = np.random.uniform(-5, 5, n)
            try_func(x)

        # Also try some specific structured points
        # Origin and axis points
        try_func(np.zeros(n))
        for d in range(n):
            for s in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                x = np.zeros(n)
                x[d] = s
                try_func(x)

        # Phase 2: Multi-start local optimization with L-BFGS-B
        # Use the best point found and several diverse starting points
        if best_f < np.inf:
            # First, refine the best point with L-BFGS-B
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Multi-start with diverse initial points
        if calls < budget - 20 and time.time() < deadline:
            n_starts = min(15, (budget - calls) // 20)
            
            for s in range(n_starts):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate diverse starting points with different scales
                if s < 3:
                    scale = 0.5 + s * 0.3
                elif s < 6:
                    scale = 1.5 + (s - 3) * 0.3
                elif s < 9:
                    scale = 2.5 + (s - 6) * 0.3
                else:
                    scale = 0.2 + (s - 9) * 0.1
                
                x0 = best_x + rng.standard_normal(n) * scale
                x0 = np.clip(x0, -5.0, 5.0)
                
                f0 = try_func(x0)
                
                # If this point is good, refine it
                if f0 < best_f + 5.0:
                    if calls < budget - 10:
                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=[(-5, 5)] * n,
                                          options={'maxiter': 80, 'gtol': 1e-8, 'ftol': 1e-12})
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = np.clip(res.x, -5.0, 5.0)
                        except Exception:
                            pass

        # Phase 3: Nelder-Mead for final polish
        if calls < budget - 30 and time.time() < deadline:
            try:
                res = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12, 'initial_simplex_size': 0.5})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: Final coordinate-wise polishing
        if calls < budget - 10 and time.time() < deadline:
            for dim_idx in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try small perturbations along this coordinate with adaptive step
                for delta in [0.01, -0.01, 0.001, -0.001, 0.0001, -0.0001, 0.00001, -0.00001]:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    x_new = best_x.copy()
                    x_new[dim_idx] += delta
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = try_func(x_new)
                    if f_new >= best_f:
                        break

        # Phase 5: Final high-precision L-BFGS-B
        if calls < budget - 5 and time.time() < deadline:
            try:
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-14})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
