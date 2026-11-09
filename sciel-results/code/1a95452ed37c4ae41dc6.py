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
        deadline = start_time + 290.0  # safety margin before 300s

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

        # --- Phase 1: Halton sequence for global exploration ---
        # Halton sequence provides low-discrepancy coverage of the search space
        def halton(index, base):
            """Generate Halton sequence value for given index and base."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    # Use the (j+1)-th prime as base for dimension j
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    points[i - 1, j] = halton(i, primes[j])
            return points

        # Generate Halton points in [0,1]^dim, then map to [-4, 4]^dim
        n_halton = 80
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts = halton_pts * 8.0 - 4.0  # map to [-4, 4]^dim

        # Also add center and some structured points
        scan_points = [np.zeros(dim)]
        # Add axis-aligned points at key locations
        for v in [-3.0, 0.0, 3.0]:
            for i in range(dim):
                pt = np.zeros(dim)
                pt[i] = v
                scan_points.append(pt)
        
        # Add Halton points
        for p in halton_pts:
            scan_points.append(p)

        # Evaluate scan points
        for p in scan_points:
            if not time_ok():
                break
            safe_func(p)

        # --- Phase 2: Local optimization from best point using L-BFGS-B ---
        x0 = best_x[0].copy()
        
        try:
            res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                           options={'maxiter': 200, 'maxls': 30, 'gtol': 1e-6})
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
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x
        except:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 4: Multiple restarts with local optimization ---
        rng = np.random.default_rng(123)
        for _ in range(10):
            if not time_ok():
                break
            x0 = rng.uniform(-4.0, 4.0, size=dim)
            try:
                res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 100, 'maxls': 20, 'gtol': 1e-6})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x
            except:
                pass

            if not time_ok():
                break

            # Quick NM polish after each restart
            try:
                res = minimize(safe_func, best_x[0].copy(), method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-11})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x
            except:
                pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
