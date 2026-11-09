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
        deadline = start_time + 280.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]

        def safe_func(x):
            if call_count[0] >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        # --- Phase 1: Halton Sequence for Global Scan ---
        # Halton sequence provides better uniform coverage than random or LHS
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Generate Halton points in [0,1]^dim, map to [-4,4]^dim
        n_halton = 60
        halton_points = np.zeros((n_halton, dim))
        bases = [2, 3, 5, 7, 11]  # first dim primes
        for i in range(1, n_halton + 1):
            for d in range(dim):
                halton_points[i - 1, d] = halton(i, bases[d])
        
        # Map [0,1] to [-4,4]
        halton_points = halton_points * 8.0 - 4.0
        
        # Add some structured points
        extra_points = [np.zeros(dim)]
        for v in [-3, 0, 3]:
            extra_points.append(np.full(dim, v))
            
        all_starts = list(halton_points) + extra_points

        # Evaluate starts
        evaluated = []
        for p in all_starts:
            if call_count[0] >= budget // 4:  # Limit scan to 1/4 of budget
                break
            if time.time() > deadline:
                break
            f = safe_func(p)
            evaluated.append((f, p.copy()))
            
        if not evaluated:
            # Fallback
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)
            return best_x, best_f

        # Sort by function value
        evaluated.sort(key=lambda x: x[0])
        
        # Take top K for local optimization
        top_k = 6
        candidates = [x[1] for x in evaluated[:top_k]]
        
        best_x = candidates[0]
        best_f = evaluated[0][0]

        # --- Phase 2: Multi-Start Local Optimization with L-BFGS-B ---
        remaining_budget = budget - call_count[0]
        reserve = 40
        alloc_per_candidate = max(15, (remaining_budget - reserve) // len(candidates))
        
        bounds = [(-5.0, 5.0)] * dim
        
        for idx, x0 in enumerate(candidates):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline:
                break
            
            # Run L-BFGS-B
            try:
                maxfun = min(alloc_per_candidate, budget - call_count[0] - reserve)
                if maxfun < 10:
                    break
                    
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'maxfun': maxfun, 'gtol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: Final Polish with Nelder-Mead ---
        if call_count[0] < budget - 10 and time.time() < deadline:
            try:
                # Nelder-Mead for final refinement
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': min(100, budget - call_count[0] - 5),
                             'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass
            
            # If still have budget, try one more L-BFGS-B pass
            if call_count[0] < budget - 5:
                try:
                    res = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': budget - call_count[0] - 3, 'gtol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
