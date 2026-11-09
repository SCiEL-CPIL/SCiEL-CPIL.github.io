import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.asarray(x, dtype=float).flatten()
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = float('inf')
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global search with a mix of structured and low-discrepancy points
        # Rosenbrock in 5D has a very narrow valley; finding the basin is crucial.
        # We use a combination of:
        # 1. Origin (0,0,0,0,0) - often near the minimum in standard Rosenbrock
        # 2. Points along the diagonal (1,1,1,1,1) scaled
        # 3. Halton sequence for uniform coverage
        # 4. A few random points for diversity
        
        candidates = []
        # Origin
        candidates.append(np.zeros(dim))
        # Diagonal points
        for scale in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(np.ones(dim) * scale)
            candidates.append(-np.ones(dim) * scale)
        
        # Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        n_halton = 20
        for i in range(1, n_halton + 1):
            p = np.array([halton(i, bases[d]) for d in range(dim)])
            p = p * 8.0 - 4.0  # Map to [-4, 4]
            candidates.append(p)
        
        # Random points
        rng = np.random.RandomState(42)
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))
        
        global_results = []
        for p in candidates:
            if calls >= budget or time.time() > deadline:
                break
            f_val = safe_call(p)
            global_results.append((f_val, p))
        
        if not global_results:
            return best_x, best_f
        
        global_results.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-start local optimization with L-BFGS-B
        # L-BFGS-B is very effective for Rosenbrock when started in the basin.
        # We try the top K candidates as starting points.
        
        top_k = min(8, len(global_results))
        
        for i in range(top_k):
            if calls >= budget - 20 or time.time() > deadline:
                break
            
            f_init, x_init = global_results[i]
            
            # Skip if we've already found a very good value and this start is much worse
            if best_f < 1e-6 and f_init > best_f * 10:
                continue
                
            try:
                res = minimize(
                    safe_call,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-15,
                        'gtol': 1e-8,
                        'maxls': 30
                    }
                )
            except Exception:
                pass
            
            # Check if we've converged well
            if best_f < 1e-10:
                break
        
        # Phase 3: If budget remains, do a final local polish around the best point
        # using a small-scale Nelder-Mead or coordinate descent
        remaining = budget - calls
        if remaining > 10 and best_f < float('inf') and time.time() < deadline:
            x0 = best_x.copy()
            
            # Try coordinate descent with adaptive step size
            step = 0.1
            improved = True
            while improved and calls < budget - 5 and time.time() < deadline:
                improved = False
                for i in range(dim):
                    for direction in [-1, 1]:
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        x_try = x0.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = safe_call(x_try)
                        if f_try < best_f:
                            x0 = x_try.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
        
        return best_x, best_f
