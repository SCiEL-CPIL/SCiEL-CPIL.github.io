"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol


class Optimizer:
    """
    Two-phase optimizer for Bent Cigar:
    Phase 1: Systematic Sobol sampling for coarse exploration.
    Phase 2: Multi-Start Powell's method for anisotropic refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        calls = 0
        best_x = None
        best_f = np.inf

        def timed_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                x = np.asarray(x, dtype=float)
                if not np.all(np.isfinite(x)):
                    return np.inf
                # Clip to box to ensure safety, though func also clips
                x_clipped = np.clip(x, -5.0, 5.0)
                v = float(func(x_clipped))
                if not np.isfinite(v):
                    return np.inf
                if v < best_f:
                    best_f = v
                    best_x = x_clipped.copy()
                return v
            except Exception:
                return np.inf

        # Phase 1: Systematic Exploration using Sobol Sequence
        # 32 points provides good coverage while leaving ample budget for refinement
        n_sobol = 32
        try:
            sampler = Sobol(d=dim, scramble=True, seed=42)
            points = sampler.random(n_sobol)
        except Exception:
            rng = np.random.default_rng(42)
            points = rng.uniform(0, 1, size=(n_sobol, dim))

        # Map from [0, 1]^d to [-5, 5]^d
        x_sobol = points * 10.0 - 5.0

        results = []
        for x in x_sobol:
            if time.time() > deadline or calls >= budget:
                break
            fx = timed_func(x)
            if np.isfinite(fx):
                results.append((fx, x))

        if not results:
            # Fallback if even initial phase fails
            x0 = np.zeros(dim)
            f0 = timed_func(x0)
            return best_x if best_x is not None else x0, best_f

        # Sort by fitness
        results.sort(key=lambda t: t[0])

        # Phase 2: Multi-Start Powell's Method
        # Powell is excellent for anisotropic functions like Bent Cigar
        # because it adapts its search directions to the principal axes.
        # We pick the top 3 distinct points to start local refinement.
        # Using 3 starts balances exploration of different basins vs depth of refinement.
        
        # Allocate remaining budget. Powell's method can be expensive in iterations 
        # but each iteration is 1 function call per direction update cycle + line search.
        # maxfev controls the total function evaluations.
        
        n_starts = 3
        remaining_budget = budget - calls
        
        # Ensure we have enough budget per start to be effective
        # Powell typically converges quickly in 5D if close to optimum.
        # 100-200 evals per start should be sufficient for high precision.
        budget_per_start = max(50, remaining_budget // n_starts)

        # Iterate over top candidates
        for i in range(n_starts):
            if i >= len(results):
                break
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            start_x = results[i][1].copy()
            
            try:
                # Use Powell's method
                # xatol and fatol set to strict tolerances to exploit remaining budget
                res = minimize(
                    timed_func,
                    start_x,
                    method='Powell',
                    options={
                        'maxiter': 100,  # Limit iterations to prevent infinite loops
                        'maxfev': budget_per_start, # Hard cap on function evals
                        'xatol': 1e-8,
                        'ftol': 1e-8,
                        'disp': False
                    }
                )
                # timed_func already updates best_x and best_f
            except Exception:
                pass

        # If best_x is still None (should not happen), return a safe default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
