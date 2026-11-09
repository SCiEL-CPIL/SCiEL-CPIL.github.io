"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve

def find_root_system(F, x0):
    # Strategy: Multi-start global search with diverse initializations + fsolve (MINPACK)
    # 1. Generate a set of diverse starting points:
    #    - x0, origin, scaled x0, random points on a sphere around x0
    # 2. For each starting point, use scipy.optimize.fsolve which uses MINPACK's
    #    hybrid Powell method (dogleg trust-region with Broyden updates).
    #    This is more robust than pure Newton and handles singular Jacobians better.
    # 3. Track the best result (lowest ||F(x)||) and return it.
    # 4. Use a call counter to stay within budget.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    call_count = [0]
    max_calls = 480  # Leave margin for safety
    
    def F_counted(x):
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise Exception("Call budget exceeded")
        return F(x)
    
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False
    
    def time_left():
        return deadline - time.time()
    
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
    
    # Generate diverse starting points
    candidates = []
    candidates.append(x0.copy())
    candidates.append(np.array([0.0, 0.0]))
    
    scale = max(1.0, norm(x0))
    
    # Scaled versions of x0
    for s in [0.5, 2.0, 0.1, 5.0]:
        candidates.append(x0 * s)
    
    # Random points on spheres of various radii around x0
    rng = np.random.default_rng(42)
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(4):
            u = rng.normal(size=2)
            u = u / norm(u)
            candidates.append(x0 + r * u)
    
    # Some points in the origin neighborhood
    for r in [0.1, 1.0, 10.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u = u / norm(u)
            candidates.append(r * u)
    
    # Deduplicate
    unique_candidates = []
    for c in candidates:
        if not np.all(np.isfinite(c)):
            continue
        is_dup = False
        for uc in unique_candidates:
            if norm(c - uc) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    
    x_best = x0.copy()
    best_norm = np.inf
    
    try:
        F_best = F(x_best)
        if np.all(np.isfinite(F_best)):
            best_norm = norm(F_best)
    except Exception:
        pass
    
    for c in unique_candidates:
        if time_left() < 2.0:
            break
        if call_count[0] > max_calls - 10:
            break
        
        try:
            # Use fsolve with MINPACK's hybrd method
            sol, info, ier, msg = fsolve(
                F_counted, 
                c, 
                full_output=True,
                maxfev=30,
                xtol=1e-14
            )
            
            if not np.all(np.isfinite(sol)):
                continue
                
            try:
                F_sol = F(sol)
                if not np.all(np.isfinite(F_sol)):
                    continue
                n_sol = norm(F_sol)
            except Exception:
                continue
            
            if n_sol < best_norm:
                best_norm = n_sol
                x_best = sol.copy()
                
            if n_sol <= 1e-8:
                return sol
                
        except Exception:
            continue
    
    # Final check
    if is_good(x_best):
        return x_best
        
    return x_best
# EVOLVE-BLOCK-END
