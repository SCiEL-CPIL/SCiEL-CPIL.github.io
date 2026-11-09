"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _halton_point(dim, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    Uses prime bases for each dimension.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1  # 1-based index for sequence
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Global-to-Local Cascade Optimizer.
    Phase 1: Broad coverage using Halton sequence points to find good basins early (AOCC boost).
    Phase 2: Refine the best few candidates using L-BFGS-B with bounds for fast convergence.
    Phase 3: High-precision polish with tight tolerances if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        
        # Track actual function calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time exceeded")
            return func(np.clip(x, -5.0, 5.0))
        
        # Phase 1: Global Exploration with Halton Sequence
        n_explore = 60
        n_explore = min(n_explore, budget - 10)
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for i in range(n_explore):
            if time.time() > deadline or call_count[0] >= budget:
                break
            
            p_unit = _halton_point(dim, i)
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = counted_func(p)
            except (ValueError, TimeoutError):
                break
            except Exception:
                continue
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            candidates.append((f, p))
        
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = counted_func(best_x)
            except Exception:
                best_f = np.inf
            candidates.append((best_f, best_x))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with L-BFGS-B (bounded, more efficient)
        bounds = [(-5.0, 5.0)] * dim
        max_starts = 6
        starts_to_try = candidates[:max_starts]
        
        for f_init, x_init in starts_to_try:
            if time.time() > deadline or call_count[0] >= budget:
                break
            
            # Skip if this start is too close to an already-refined point
            # (simple deduplication)
            skip = False
            for _, x_prev in starts_to_try:
                if np.linalg.norm(x_init - x_prev) < 0.1:
                    skip = True
                    break
            if skip:
                continue
            
            try:
                res = minimize(
                    counted_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except (ValueError, TimeoutError):
                break
            except Exception:
                continue

        # Phase 3: High-precision polish from the best point
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                res2 = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
                )
                
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
                    
            except (ValueError, TimeoutError):
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
