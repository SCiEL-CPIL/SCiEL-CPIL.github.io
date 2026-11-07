"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local optimizer with adaptive step sizes for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return best_f
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            else:
                x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse multi-start with local polishing
        # Use several starting points spread across the domain
        n_starts = min(10, max(3, budget // 50))
        
        # Initial broad exploration
        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Deterministic + random starting points
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.ones(dim) * 2.5
            elif i == 2:
                x0 = np.ones(dim) * -2.5
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            f0 = safe_func(x0)
            
            # Local optimization using Nelder-Mead with adaptive parameters
            # Use limited iterations to save budget for other starts
            max_iter_local = min(30, max(5, (budget - call_count) // n_starts))
            
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_local,
                        'xatol': 0.05,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.success and np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 2: Fine-grained local search around best point
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            # Use a smaller step size search
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Adaptive step: shrink if stuck
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                    if sigma < 0.001:
                        break

        # Phase 3: Final polish with very small steps
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            sigma = 0.01
            x = best_x.copy()
            fx = best_f
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                    if sigma < 0.0001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
