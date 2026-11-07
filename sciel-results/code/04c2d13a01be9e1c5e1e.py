"""Combined (1+1)-ES with Nelder-Mead polish for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive sigma followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility in evaluation
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Track global best
        best_x = None
        best_f = np.inf
        total_calls = 0

        def record_call(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 0: Initialization with multiple random starts to ensure good start point
        # Bent Cigar is unimodal but shifted/rotated. A good start is crucial.
        # We evaluate 5 random points and the origin.
        init_candidates = [np.zeros(dim)]
        for _ in range(4):
            init_candidates.append(rng.uniform(-5, 5, dim))
        
        for cand in init_candidates:
            if total_calls >= budget or time.time() > deadline:
                break
            f_cand = func(cand)
            total_calls += 1
            record_call(cand, f_cand)

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        # Phase 1: (1+1)-ES
        # Reserve budget for Nelder-Mead polish
        # NM in dim=5 needs ~dim+1 = 6 points per iteration. 
        # With 100 calls, we can do ~15-20 iterations, which is plenty for polish.
        polish_budget = min(150, budget // 4) 
        es_budget = budget - polish_budget
        
        x = best_x.copy()
        fx = best_f
        
        # (1+1)-ES Parameters
        # Initial sigma: large enough to explore, small enough to converge.
        # For dim=5, a sigma of 2.0-3.0 is reasonable.
        sigma = 2.5
        success_rate = 1/5
        
        while total_calls < es_budget and time.time() < deadline:
            # Generate candidate
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            total_calls += 1
            
            if fy < fx:
                # Success
                x, fx = y, fy
                record_call(x, fx)
                sigma *= 1.15  # Slightly more aggressive growth
            else:
                # Failure
                sigma *= 0.90  # Slightly more aggressive shrinkage

            # Clamp sigma
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-10:
                sigma = 1e-10

        # Phase 2: Nelder-Mead Polish
        remaining_budget = budget - total_calls
        if remaining_budget > 10 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                nonlocal total_calls
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                total_calls += 1
                # Clip to box to ensure validity
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                if val < best_f:
                    best_f = val
                    best_x = z_clipped.copy()
                return val
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget, 
                        'xatol': 1e-10, 
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
