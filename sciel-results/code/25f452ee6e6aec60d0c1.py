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

        # --- Phase 1: Global Scan with Latin Hypercube Sampling ---
        # LHS ensures better coverage of the [-4, 4]^5 box than pure random or grid
        n_scan = 40
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # Split [0, 1] into n_scan intervals, pick one random point in each, map to [-4, 4]
        edges = np.linspace(0, 1, n_scan + 1)
        lhs_samples = np.zeros((n_scan, dim))
        for d in range(dim):
            # For each dimension, shuffle the intervals to decorrelate
            perm = rng.permutation(n_scan)
            for i in range(n_scan):
                # Random point within the interval
                lower = edges[perm[i]]
                upper = edges[perm[i] + 1]
                lhs_samples[i, d] = rng.uniform(lower, upper)
        
        # Map [0, 1] to [-4, 4]
        lhs_samples = lhs_samples * 8.0 - 4.0
        
        # Add origin and some structured points to ensure we don't miss obvious basins
        extra_points = [np.zeros(dim)]
        for v in [-3, 3]:
            extra_points.append(np.full(dim, v))
            
        all_starts = list(lhs_samples) + extra_points

        # Evaluate starts
        evaluated = []
        for p in all_starts:
            if call_count[0] >= budget // 3: # Limit scan to 1/3 of budget
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
        top_k = 5
        candidates = [x[1] for x in evaluated[:top_k]]
        
        best_x = candidates[0]
        best_f = evaluated[0][0]

        # --- Phase 2: Multi-Start Local Optimization ---
        # Allocate remaining budget among the top candidates
        remaining_budget = budget - call_count[0]
        # Reserve some for final polish
        reserve = 50
        alloc_per_candidate = max(10, (remaining_budget - reserve) // len(candidates))
        
        bounds = [(-5.0, 5.0)] * dim
        
        for idx, x0 in enumerate(candidates):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline:
                break
            
            # Run L-BFGS-B
            try:
                # Limit function evaluations for this specific run
                maxfun = min(alloc_per_candidate, budget - call_count[0] - reserve)
                if maxfun < 10:
                    break
                    
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': maxfun}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # Use Nelder-Mead or L-BFGS-B with high precision on the best point
        if call_count[0] < budget - 5 and time.time() < deadline:
            try:
                # L-BFGS-B again with tighter tolerance if possible
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': budget - call_count[0] - 5, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # If still have budget, try a few random perturbations and NM
            if call_count[0] < budget - 20:
                for _ in range(3):
                    if call_count[0] >= budget - 10:
                        break
                    if time.time() > deadline:
                        break
                    # Small perturbation
                    step = 0.1
                    x_try = best_x + rng.normal(0, step, size=dim)
                    x_try = np.clip(x_try, -5.0, 5.0)
                    # Simple Nelder-Mead for a few iterations
                    try:
                        res_nm = minimize(
                            safe_func,
                            x_try,
                            method='Nelder-Mead',
                            options={'maxiter': 50, 'maxfev': 20}
                        )
                        if res_nm.fun < best_f:
                            best_f = res_nm.fun
                            best_x = res_nm.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
