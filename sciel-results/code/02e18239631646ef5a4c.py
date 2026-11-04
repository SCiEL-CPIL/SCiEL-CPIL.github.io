import numpy as np
from scipy.optimize import minimize
import time
import math

N = 50
DH = 1.0 / 20.0
OFFSET = 10

# Wall-clock deadline with safety margin
_DEADLINE = 115.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _start

def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out

def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)

def _smooth_max(x, sigma=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    # Avoid underflow/overflow in exp
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_log(z):
    """
    Objective to minimize: -log(Ratio).
    z is a real vector of length N. h = exp(z).
    We normalize h to have L1 norm 1 to handle scale invariance, 
    or we can just let the optimizer find the scale. 
    However, log-sum-exp is numerically stable if we shift.
    """
    # Softplus-like or just exp? 
    # h = exp(z) can overflow. 
    # Let's shift z to keep h in a reasonable range.
    z_shifted = z - np.mean(z)
    h = np.exp(z_shifted)
    
    # Normalize to L1 = 1 to fix scale
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e12
    h = h / l1
    
    l2sq = np.sum(h**2)
    
    # Compute autoconvolution
    ac = _autoconv(h)
    
    # Use smooth max for differentiability
    # Adaptive sigma? Start small.
    peak = _smooth_max(ac, sigma=0.005)
    
    if peak < 1e-12:
        return 1e12
        
    ratio = l2sq / peak
    if ratio <= 0:
        return 1e12
        
    return -math.log(ratio)

def _optimize_log(z0, time_limit):
    """Run L-BFGS-B optimization on log parameters."""
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_log,
            z0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 1000}
        )
    except Exception:
        return None

    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
        
    z_opt = result.x
    # Recover h
    z_shifted = z_opt - np.mean(z_opt)
    h_opt = np.exp(z_shifted)
    h_opt = h_opt / np.sum(h_opt)
    
    return h_opt

def _sparsify_and_optimize(h, K, time_limit):
    """Keep top K elements, set others to 0, and re-optimize the support and weights."""
    if K >= N:
        return h, _ratio(h)
        
    # Find top K indices
    indices = np.argsort(h)[-K:]
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    h_sparse = np.zeros(N)
    h_sparse[indices] = h[indices]
    
    if np.sum(h_sparse) == 0:
        return h, 0.0
        
    # Initialize z from h_sparse
    # For zero elements, set z to a large negative value (e.g., -10)
    # For non-zero, z = log(h)
    # We only optimize the non-zero ones? 
    # L-BFGS-B optimizes all. If we fix zeros, we should use bounds or just optimize the subset.
    # Let's optimize the subset directly.
    
    h_sub = h_sparse[indices]
    # Normalize
    h_sub = h_sub / np.sum(h_sub)
    
    # z_sub = log(h_sub)
    # Add small noise to avoid log(0) if any are exactly 0 (shouldn't happen if top K)
    h_sub = np.maximum(h_sub, 1e-10)
    z_sub = np.log(h_sub)
    
    # Define local objective
    def obj_sub(z_sub):
        h_sub_val = np.exp(z_sub - np.mean(z_sub))
        h_sub_val = h_sub_val / np.sum(h_sub_val)
        
        # Reconstruct full h
        h_full = np.zeros(N)
        h_full[indices] = h_sub_val
        
        l2sq = np.sum(h_full**2)
        ac = _autoconv(h_full)
        peak = _smooth_max(ac, sigma=0.005)
        if peak < 1e-12:
            return 1e12
        ratio = l2sq / peak
        if ratio <= 0:
            return 1e12
        return -math.log(ratio)
        
    try:
        res = minimize(
            obj_sub,
            z_sub,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-9, 'maxfun': 500}
        )
    except Exception:
        return h_sparse, _ratio(h_sparse)
        
    z_opt_sub = res.x
    h_opt_sub = np.exp(z_opt_sub - np.mean(z_opt_sub))
    h_opt_sub = h_opt_sub / np.sum(h_opt_sub)
    
    h_opt_full = np.zeros(N)
    h_opt_full[indices] = h_opt_sub
    
    return h_opt_full, _ratio(h_opt_full)

def construct():
    global _start
    _start = time.time()

    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Sparse Spikes
    for K in [5, 10, 15, 20]:
        indices = rng.choice(N, K, replace=False)
        h_sp = np.zeros(N)
        h_sp[indices] = 1.0
        initializations.append(h_sp)
        
    # 2. Uniform
    initializations.append(np.ones(N))
    
    # 3. Gaussian
    x = np.linspace(-0.25, 0.25, N)
    for w in [0.02, 0.05, 0.1]:
        h_g = np.exp(-x**2 / (2*w**2))
        initializations.append(h_g)
        
    # 4. Random Smooth
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        h_raw = rng_local.random(N)
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h_smooth = np.convolve(h_raw, kernel, mode='same')
        initializations.append(h_smooth)

    # Sort initializations
    scored_inits = []
    for h0 in initializations:
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) == 0: continue
        h0 = h0 / np.sum(h0)
        score = _ratio(h0)
        scored_inits.append((score, h0))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    time_per_opt = 5.0
    
    # Phase 1: Optimize from initializations
    for score, h0 in scored_inits[:10]:
        if _time_left() < time_per_opt + 2.0:
            break
            
        # Convert h0 to z0 (log space)
        h0_safe = np.maximum(h0, 1e-10)
        z0 = np.log(h0_safe)
        
        h_opt = _optimize_log(z0, time_per_opt)
        
        if h_opt is not None:
            h_opt = np.maximum(h_opt, 0.0)
            score_opt = _ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()

    # Phase 2: Sparsification Polish
    # Try to improve the best_h by sparsifying it
    if _time_left() > 20.0:
        for K in [10, 15, 20, 25, 30]:
            if _time_left() < 8.0:
                break
            h_sp, score_sp = _sparsify_and_optimize(best_h, K, 6.0)
            if score_sp > best_score:
                best_score = score_sp
                best_h = h_sp.copy()
                
            # Also try sparsifying a random good candidate if best_h is stuck?
            # No, just iterate on the current best.

    # Final safety check
    if best_h is None or not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]

if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
