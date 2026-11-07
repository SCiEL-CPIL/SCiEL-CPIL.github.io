"""Optimizer using SLSQP with multi-start and adaptive refinement for BBOB f8."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start SLSQP optimizer with diverse exploration for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                return best_f
            call_count[0] += 1
            try:
                val = float(func(x))
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val
                else:
                    return 1e10
            except Exception:
                return 1e10

        def time_ok():
            return time.time() < deadline

        # Phase 1: Diverse coarse sampling to find good starting regions
        # Use structured points first, then random
        n_coarse = 15
        rng = np.random.default_rng(42)

        candidates = []

        for i in range(n_coarse):
            if not time_ok() or call_count[0] >= budget - 20:
                break
            
            # Structured points first, then random
            if i < dim:
                x0 = np.zeros(dim)
                x0[i] = 2.0
            elif i == dim:
                x0 = np.zeros(dim)
            elif i == dim + 1:
                x0 = np.ones(dim)
            elif i == dim + 2:
                x0 = -np.ones(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            f0 = wrapped_func(x0)
            candidates.append((f0, x0.copy()))

        # Sort candidates by function value
        candidates.sort(key=lambda c: c[0])

        # Keep top diverse seeds (at most 6)
        max_seeds = 6
        if len(candidates) > max_seeds:
            top = [candidates[0]]
            for c in candidates[1:]:
                if len(top) >= max_seeds:
                    break
                diverse = True
                for tc in top:
                    if np.linalg.norm(c[1] - tc[1]) < 1.5:
                        diverse = False
                        break
                if diverse:
                    top.append(c)
            candidates = top

        # Phase 2: Intensive local refinement using SLSQP from diverse starts
        for f0, x0 in candidates:
            if not time_ok() or call_count[0] >= budget - 15:
                break
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-14, 'maxfun': 80}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Multi-scale refinement around the best point
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            scales = [1.0, 0.5, 0.25, 0.1]
            for scale in scales:
                if not time_ok() or call_count[0] >= budget - 10:
                    break
                
                # Generate perturbation around best point
                perturbation = rng.normal(0, scale, dim)
                start_point = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    res = minimize(
                        wrapped_func,
                        start_point,
                        method='SLSQP',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 80, 'ftol': 1e-14, 'maxfun': 50}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: Final high-precision polish from best point
        if best_x is not None and time_ok() and call_count[0] < budget - 5:
            try:
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'maxfun': 60}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
