"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _ratio(h):
    """Compute (sum h_i^2) / (sum h_i * max autocorrelation)."""
    h = np.asarray(h, dtype=np.float64)
    if h.sum() == 0:
        return 0.0
    l2_sq = np.dot(h, h)
    l1 = h.sum()
    # Full autocorrelation
    ac = np.correlate(h, h, mode='full')
    sup_ac = np.max(ac)
    if sup_ac == 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _elapsed():
    return time.time() - _START_TIME

def _optimize_coordinate(h, i):
    """
    Optimize the i-th coordinate of h, holding others fixed.
    Uses scipy.optimize.minimize_scalar on the 1D slice.
    Returns the new h with updated i-th coordinate.
    """
    h = np.array(h, dtype=np.float64)
    h_fixed = h.copy()
    h_fixed[i] = 0.0
    
    def obj_1d(x):
        if x < 0:
            return 1e10
        h_trial = h_fixed.copy()
        h_trial[i] = x
        if h_trial.sum() == 0:
            return 1e10
        return -_ratio(h_trial)
    
    # Determine bounds for x
    # Scale invariant shape search, but we need a bound.
    # Use current max * 2 or 10 as upper bound.
    current_max = np.max(h)
    if current_max == 0:
        current_max = 1.0
    upper_bound = max(10.0 * current_max, 1.0)
    
    try:
        res = minimize_scalar(obj_1d, bounds=(0, upper_bound), method='bounded', options={'maxiter': 100})
        if res.success:
            h[i] = res.x
    except Exception:
        pass
        
    return h

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    # Initialize with diverse candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Delta (center)
    d1 = np.zeros(N); d1[N//2] = 1.0
    candidates.append(d1)
    
    # 3. Delta (edge)
    d2 = np.zeros(N); d2[0] = 1.0
    candidates.append(d2)
    
    # 4. Gaussian
    x = np.linspace(-1, 1, N)
    candidates.append(np.exp(-x**2 * 2))
    
    # 5. Random
    candidates.append(rng.random(N))
    
    # 6. Random sparse
    h_sparse = rng.random(N)
    h_sparse[rng.random(N) < 0.5] = 0
    candidates.append(h_sparse)
    
    for h_init in candidates:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Normalize to L2=1 to fix scale during optimization
        norm = np.linalg.norm(h_init)
        if norm < 1e-10:
            continue
        h_curr = h_init / norm
        
        # Coordinate Descent
        for _round in range(10):
            if _elapsed() > _DEADLINE - 10:
                break
            for i in range(N):
                if _elapsed() > _DEADLINE - 10:
                    break
                h_curr = _optimize_coordinate(h_curr, i)
            
        # Re-normalize to L2=1 after optimization
        norm = np.linalg.norm(h_curr)
        if norm > 1e-10:
            h_curr = h_curr / norm
            
        val = _ratio(h_curr)
        if val > best_val:
            best_val = val
            best_h = h_curr
    
    # Return best_h. 
    # If evaluator normalizes, scale doesn't matter.
    # If evaluator doesn't, L2=1 is a safe standard scale.
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
