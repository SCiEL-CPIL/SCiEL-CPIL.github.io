"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]
        best_x = [np.zeros(dim)]
        best_f = [np.inf]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f = float(func(x))
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()
            return f

        def time_ok():
            return call_count[0] < budget and time.time() < deadline

        # --- Phase 1: Global scan ---
        # Use a mix of structured points to find a good starting region
        scan_points = []
        
        # Center
        scan_points.append(np.zeros(dim))
        
        # Axis-aligned points
        for v in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            for i in range(dim):
                pt = np.zeros(dim)
                pt[i] = v
                scan_points.append(pt)
        
        # Corner-like points (subset)
        # 2^5 = 32 points at {-3, 3}
        for vals in np.ndindex(2, 2, 2, 2, 2):
            pt = np.array([3.0 if v else -3.0 for v in vals], dtype=float)
            scan_points.append(pt)
        
        # Some random points
        rng = np.random.default_rng(42)
        for _ in range(20):
            pt = rng.uniform(-4.0, 4.0, size=dim)
            scan_points.append(pt)

        # Evaluate scan points
        for p in scan_points:
            if not time_ok():
                break
            safe_func(p)

        # --- Phase 2: Local optimization from best point ---
        x0 = best_x[0].copy()
        
        # Try L-BFGS-B first
        try:
            res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                           options={'maxiter': 100, 'maxls': 20})
            if res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x
        except:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 3: Nelder-Mead polish ---
        x0 = best_x[0].copy()
        try:
            res = minimize(safe_func, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x
        except:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 4: Multiple restarts if budget remains ---
        # If we still have budget, do a few random restarts with local optimization
        rng = np.random.default_rng(123)
        for _ in range(5):
            if not time_ok():
                break
            x0 = rng.uniform(-4.0, 4.0, size=dim)
            try:
                res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 50, 'maxls': 10})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x
            except:
                pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
