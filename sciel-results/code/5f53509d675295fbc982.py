"""Optimized BBOB f21 (Gallagher 101 peaks) solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Global search via differential evolution + local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Quick deterministic evaluations for early AOCC progress
        # Evaluate a small set of strategic points to get initial bounds fast
        strategic_points = [
            np.zeros(n),
            np.array([4.0] + [0.0]*(n-1)),
            np.array([-4.0] + [0.0]*(n-1)),
            np.array([2.0] + [0.0]*(n-1)),
            np.array([-2.0] + [0.0]*(n-1)),
            np.array([0.5]*n),
            np.array([-0.5]*n),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0][:n]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:n]),
        ]
        
        results = []
        for pt in strategic_points:
            if calls >= budget or not time_ok():
                break
            x = np.array(pt, dtype=np.float64)
            f = safe_call(x)
            if np.isfinite(f):
                results.append((f, x.copy()))
        
        if not results:
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            return x0, f0
        
        results.sort(key=lambda item: item[0])
        best_x = results[0][1].copy()
        best_f = results[0][0]
        
        # Phase 2: Global search with differential evolution (handles multimodality)
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 50:
                # Use DE with bounded domain [-4, 4]^n (minimizer is in this range)
                bounds = [(-4.0, 4.0)] * n
                maxiter = max(1, (remaining - 10) // (n + 1))  # rough estimate
                try:
                    res = minimize(
                        safe_call,
                        best_x,
                        method='differential_evolution',
                        bounds=bounds,
                        options={
                            'maxiter': maxiter,
                            'popsize': max(5, min(20, (remaining - 10) // n)),
                            'tol': 1e-8,
                            'init': 'latinhypercube',
                            'seed': 42,
                            'polish': True
                        }
                    )
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Local refinement with Nelder-Mead from best global result
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 20:
                try:
                    res2 = minimize(
                        safe_call, best_x, method='Nelder-Mead',
                        options={'maxfev': min(remaining, 100), 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res2.fun < best_f and np.isfinite(res2.fun):
                        best_f = res2.fun
                        best_x = res2.x.copy()
                except Exception:
                    pass
        
        # Phase 4: Fine local search if budget remains
        if calls < budget and time_ok():
            remaining = budget - calls
            if remaining > 5:
                rng = np.random.RandomState(42)
                for _ in range(min(remaining // 3, 30)):
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.randn(n) * 0.05
                    x_try = best_x + perturbation
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = safe_call(x_try)
                    if np.isfinite(f_try) and f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
