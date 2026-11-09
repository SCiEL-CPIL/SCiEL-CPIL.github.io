"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Nelder-Mead simplex refinement with (1+1)-ES polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin below 300 s
        n = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        # Track calls and best
        calls_used = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Build diverse starting points
        starts = []
        starts.append(np.zeros(n))
        
        # Halton sequence for quasi-random coverage
        def halton(idx, base):
            f, r = 1.0, 0.0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r

        for k in range(1, 15):
            pt = np.array([halton(k, b) for b in range(2, n + 2)])
            starts.append(lo + pt * (hi - lo))
        
        # Add some random points for diversity
        for _ in range(5):
            starts.append(rng.uniform(lo, hi, n))

        # Evaluate all starting points
        for s in starts:
            if calls_used >= budget or time.time() > deadline:
                break
            fs = func(s)
            record(s, fs)

        if best_x is None:
            return np.zeros(n), np.inf

        # Phase 2: Nelder-Mead refinement on the best point found
        # Use a wrapper to count function calls
        nm_calls = 0
        nm_budget = min(budget - calls_used, 200)  # Limit NM calls to avoid getting stuck
        
        def nm_func(x):
            nonlocal nm_calls
            if time.time() > deadline:
                return 1e10
            x_clipped = np.clip(x, lo, hi)
            val = func(x_clipped)
            nm_calls += 1
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Set up Nelder-Mead options
        nm_options = {
            'maxiter': nm_budget,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }

        try:
            result = minimize(
                nm_func,
                best_x.copy(),
                method='Nelder-Mead',
                options=nm_options
            )
            # Record the result if it's better
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, lo, hi).copy()
        except Exception:
            pass

        # Phase 3: Polish with (1+1)-ES if we have budget left
        if best_x is not None and calls_used < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            sigma = 0.5
            
            while calls_used < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), lo, hi)
                fy = func(y)
                calls_used += 1
                
                if fy < f:
                    x, f = y, fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 5.0)
                else:
                    sigma = max(sigma * 0.9, 1e-4)
                    
                # If we get stuck, try a small jump
                if sigma < 1e-3:
                    x = np.clip(x + 0.1 * rng.standard_normal(n), lo, hi)
                    f = func(x)
                    calls_used += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
