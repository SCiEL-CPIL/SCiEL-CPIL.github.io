import numpy as np
import time
from scipy.optimize import minimize

N = 100
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _max_overlap_value(h):
    """Return the maximum overlap value."""
    corrs = _compute_overlaps(h)
    return np.max(corrs)

def _project_to_feasible(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(30):
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        n_active = np.sum(mask)
        if n_active == 0:
            break
        step = diff / n_active
        new_h = h.copy()
        new_h[mask] += step
        if np.any(new_h < -1e-12) or np.any(new_h > 1 + 1e-12):
            lo, hi = -1.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                test_h = np.clip(h + mid, 0, 1)
                test_sum = np.sum(test_h)
                if test_sum > target_sum:
                    hi = mid
                else:
                    lo = mid
            h = np.clip(h + (lo + hi) / 2, 0, 1)
        else:
            h = new_h
        diff = target_sum - np.sum(h)
    
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _beatty_sequence(alpha, beta, n):
    """
    Generate a Beatty/Kronecker sequence.
    h[i] = 1 if {alpha*i + beta} < 0.5 else 0
    This gives a sequence with mean ~0.5 and bounded autocorrelation deviations
    due to equidistribution of quadratic irrationals.
    """
    idx = np.arange(n)
    vals = np.mod(alpha * idx + beta, 1.0)
    h = (vals < 0.5).astype(float)
    return h

def _golden_ratio_candidates(n):
    """
    Generate candidate sequences using golden ratio and related quadratic irrationals.
    The golden ratio phi = (1+sqrt(5))/2 has the best Diophantine approximation constant,
    giving the most uniform distribution.
    """
    sqrt5 = np.sqrt(5.0)
    phi = (1.0 + sqrt5) / 2.0  # golden ratio ~1.618
    alpha_golden = 1.0 / phi  # ~0.618, the fractional part
    
    candidates = []
    
    # Primary: golden ratio with various offsets
    for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        h = _beatty_sequence(alpha_golden, beta, n)
        candidates.append(h)
    
    # Secondary: other quadratic irrationals
    # 1/sqrt(2) ~ 0.707
    alpha_sqrt2 = 1.0 / np.sqrt(2.0)
    for beta in [0.0, 0.25, 0.5, 0.75]:
        h = _beatty_sequence(alpha_sqrt2, beta, n)
        candidates.append(h)
    
    # (sqrt(2)-1) ~ 0.414
    alpha_sqrt2m1 = np.sqrt(2.0) - 1.0
    for beta in [0.0, 0.25, 0.5, 0.75]:
        h = _beatty_sequence(alpha_sqrt2m1, beta, n)
        candidates.append(h)
    
    # (sqrt(3)-1)/2 ~ 0.366
    alpha_sqrt3 = (np.sqrt(3.0) - 1.0) / 2.0
    for beta in [0.0, 0.25, 0.5, 0.75]:
        h = _beatty_sequence(alpha_sqrt3, beta, n)
        candidates.append(h)
    
    return candidates

def _soft_min_polish(h, temps=[0.1, 0.05, 0.02, 0.01]):
    """Polish using soft-min (log-sum-exp) approximation."""
    n = len(h)
    
    def _obj_and_grad(h_vec, temp):
        h_vec = np.clip(h_vec, 0, 1)
        corrs = _compute_overlaps(h_vec)
        shifted = corrs / temp
        max_sh = np.max(shifted)
        logsumexp = max_sh + np.log(np.sum(np.exp(shifted - max_sh)))
        obj = temp * logsumexp
        
        weights = np.exp(shifted - max_sh)
        weights /= weights.sum()
        
        grad = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1]
            if w < 1e-10:
                continue
            for j in range(n - k):
                grad[j] += w * h_vec[j + k]
                grad[j + k] += w * h_vec[j]
            grad *= (2.0 / n)
        
        return obj, grad
    
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    for temp in temps:
        if _time_remaining() < 10.0:
            break
        try:
            res = minimize(
                _obj_and_grad,
                h,
                args=(temp,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.0, 1.0)] * n,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 30, 'maxfun': 1000}
            )
            h_new = _project_to_feasible(res.x, n / 2.0)
            v_new = _max_overlap_value(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
            h = h_new
        except Exception:
            pass
    
    return best_h, best_val

def _refine_binary(h_binary):
    """
    Take a binary Beatty sequence and refine it to non-binary values
    while maintaining the structural properties.
    """
    n = len(h_binary)
    h = h_binary.copy()
    
    # First, adjust to exact sum = n/2
    h = _project_to_feasible(h, n / 2.0)
    
    # Now try to relax binary constraints locally
    # Identify positions where small perturbations might help
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    # Try local adjustments: for each position, try small changes
    # that reduce the maximum correlation
    for iteration in range(50):
        if _time_remaining() < 5.0:
            break
            
        # Find which shifts are near-max
        threshold = max_val - 1e-4
        active_shifts = np.where(corrs >= threshold)[0] + 1
        
        # Try perturbing each element slightly
        improved = False
        for i in range(n):
            old_val = h[i]
            # Try a few perturbations
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                h_test = h.copy()
                h_test[i] = np.clip(old_val + delta, 0, 1)
                h_test = _project_to_feasible(h_test, n / 2.0)
                val_test = _max_overlap_value(h_test)
                if val_test < max_val - 1e-6:
                    h = h_test
                    max_val = val_test
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return h

def construct():
    t0 = time.time()
    target_sum = N / 2.0
    
    # Generate Beatty sequence candidates
    candidates = _golden_ratio_candidates(N)
    
    best_h = None
    best_val = np.inf
    
    # Evaluate each candidate and keep the best
    for h_cand in candidates:
        if _time_remaining() < 10.0:
            break
        
        # Project to feasible region
        h_feasible = _project_to_feasible(h_cand, target_sum)
        val = _max_overlap_value(h_feasible)
        
        if val < best_val:
            best_val = val
            best_h = h_feasible.copy()
    
    # Refine the best binary sequence
    if best_h is not None and _time_remaining() > 20.0:
        h_refined = _refine_binary(best_h)
        val_refined = _max_overlap_value(h_refined)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
    
    # Polish with soft-min optimization
    if _time_remaining() > 15.0:
        h_polished, val_polished = _soft_min_polish(best_h, temps=[0.2, 0.1, 0.05, 0.02, 0.01])
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
    
    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h, target_sum)
    
    return best_h

# EVOLVE-BLOCK-END
