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
        calls = [0]
        best_x = [np.zeros(dim)]
        best_f = [float('inf')]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls[0] >= budget or time.time() > deadline:
                return float('inf')
            calls[0] += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        # --- Phase 1: 1D Curve Search (Lissajous/Knots) ---
        # Structural Change: Reduce global search to 1D parameter optimization
        # over several winding curves that traverse the [-4, 4]^dim box.
        
        rng = np.random.default_rng(42)
        
        # Generate a set of incommensurate frequencies for Lissajous curves
        # We use several different "knots" to increase the chance of hitting the valley
        num_curves = 5
        curve_results = []
        
        for c in range(num_curves):
            if not time_ok():
                break
            
            # Frequencies should be integers or simple rationals to ensure coverage, 
            # but incommensurate is better for space filling in finite time.
            # We use random integers to create unique winding patterns.
            freqs = rng.integers(1, 10, size=dim)
            phases = rng.uniform(0, 2*np.pi, size=dim)
            
            def curve_map(t, freqs=freqs, phases=phases):
                # t is in [0, 1]
                # Map to angles
                angles = 2 * np.pi * (freqs * t + phases)
                # Map to [-4, 4]
                return 4.0 * np.sin(angles)
            
            def curve_func(t):
                if not time_ok():
                    return float('inf')
                x = curve_map(t)
                return safe_func(x)
            
            try:
                # Use Brent's method for 1D minimization
                # Bounds [0, 1]
                res_1d = minimize_scalar(curve_func, bounds=(0.0, 1.0), method='brent', options={'xatol': 1e-4})
                if np.isfinite(res_1d.fun):
                    x_best_curve = curve_map(res_1d.x)
                    # Re-evaluate to ensure it's in our best tracking (safe_func already did this, 
                    # but we want the specific point for local restarts)
                    # Note: safe_func updates best_x/best_f if it finds a better point.
                    # We just need to collect these candidates for the next phase.
                    curve_results.append((res_1d.fun, x_best_curve))
            except Exception:
                pass
        
        # Also add the origin and a few random points as fallbacks if curves fail
        if not time_ok():
            return best_x[0], best_f[0]
            
        fallback_pts = [np.zeros(dim)]
        for _ in range(3):
            fallback_pts.append(rng.uniform(-4.0, 4.0, size=dim))
            
        global_results = curve_results
        for p in fallback_pts:
            if not time_ok():
                break
            f = safe_func(p)
            if np.isfinite(f):
                global_results.append((f, p.copy()))
                
        if not global_results:
            return best_x[0], best_f[0]

        # Sort by function value
        global_results.sort(key=lambda x: x[0])

        # --- Phase 2: L-BFGS-B on top candidates ---
        # Rosenbrock benefits greatly from gradient-based local optimization
        top_k = min(6, len(global_results))

        for i in range(top_k):
            if not time_ok() or calls[0] >= budget - 20:
                break

            f_init, x_init = global_results[i]

            try:
                res = minimize(
                    safe_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 80,
                        'ftol': 1e-14,
                        'gtol': 1e-8,
                        'maxls': 15
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x.copy()
            except Exception:
                pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 3: Nelder-Mead polish on the best point ---
        x0 = best_x[0].copy()
        try:
            res = minimize(
                safe_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if np.isfinite(res.fun) and res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x.copy()
        except Exception:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 4: Additional restarts if budget remains ---
        rng2 = np.random.default_rng(123)
        remaining_budget = budget - calls[0]
        if remaining_budget > 50:
            n_restarts = min(4, remaining_budget // 40)
            for _ in range(n_restarts):
                if not time_ok() or calls[0] >= budget - 10:
                    break
                x0 = rng2.uniform(-4.0, 4.0, size=dim)
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 50,
                            'ftol': 1e-12,
                            'gtol': 1e-6,
                            'maxls': 10
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f[0]:
                        best_f[0] = res.fun
                        best_x[0] = res.x.copy()
                except Exception:
                    pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
