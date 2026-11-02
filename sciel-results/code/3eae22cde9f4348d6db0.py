import numpy as np
import time
from scipy.optimize import differential_evolution

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
        # Check if uniform step would violate bounds
        new_h = h.copy()
        new_h[mask] += step
        if np.any(new_h < -1e-12) or np.any(new_h > 1 + 1e-12):
            # Use binary search for the right offset
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
    
    # Final exact adjustment with careful handling
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute among elements not at bounds
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _spectral_init():
    """Deterministic circulant spectral construction as initial guess."""
    EPS = 1e-4
    A = max(0.25 * N - (N - 3) * EPS, 0.0)
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(np.fft.ifft(H))
    return _project_to_feasible(h, N / 2.0)

def _de_objective(x, target_sum):
    """Objective function for Differential Evolution."""
    h = _project_to_feasible(x, target_sum)
    val = _max_overlap_value(h)
    # Add a small penalty if projection failed to meet sum exactly (unlikely but safe)
    sum_err = abs(np.sum(h) - target_sum)
    if sum_err > 1e-9:
        val += sum_err * 1000.0
    return val

def _differential_evolution_optimize(h_init, max_iters=100, popsize=15):
    """Optimize using Differential Evolution."""
    n = len(h_init)
    target_sum = n / 2.0
    bounds = [(0.0, 1.0)] * n
    
    # Use the spectral init as one of the seeds or just let DE explore
    # DE with bounds is robust.
    
    # We can provide a strategy to seed with h_init?
    # differential_evolution doesn't take a seed vector directly, but we can
    # initialize the population manually if we used a lower-level API, 
    # but scipy's DE is high-level. 
    # We'll just run DE. It's global.
    
    try:
        res = differential_evolution(
            _de_objective,
            bounds=bounds,
            args=(target_sum,),
            maxiter=max_iters,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=False # We will polish later or use the raw result
        )
        h_best = _project_to_feasible(res.x, target_sum)
        val_best = _max_overlap_value(h_best)
        return h_best, val_best
    except Exception:
        return h_init, _max_overlap_value(h_init)

def _local_polish(h):
    """Quick local polish using L-BFGS-B on the soft-min objective."""
    from scipy.optimize import minimize
    
    n = len(h)
    target_sum = n / 2.0
    
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
    
    # Short polish
    try:
        res = minimize(
            _obj_and_grad,
            h,
            args=(0.05,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(0.0, 1.0)] * n,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxls': 20, 'maxfun': 200}
        )
        h_new = _project_to_feasible(res.x, target_sum)
        v_new = _max_overlap_value(h_new)
        if v_new < best_val:
            best_val = v_new
            best_h = h_new.copy()
    except Exception:
        pass
        
    return best_h, best_val

def construct():
    t0 = time.time()
    target_sum = N / 2.0
    
    # Start with spectral initialization
    h = _spectral_init()
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    # Phase 1: Differential Evolution (Global Search)
    if _time_remaining() > 60.0:
        # Adjust iterations/popsize based on time
        # DE is slower per iteration than gradient steps, but global
        max_iters = 50
        popsize = 10
        if _time_remaining() > 100.0:
            max_iters = 100
            popsize = 15
            
        h1, val1 = _differential_evolution_optimize(h, max_iters=max_iters, popsize=popsize)
        if val1 < best_val:
            best_val = val1
            best_h = h1.copy()
    
    # Phase 2: Local Polish (Exploit)
    if _time_remaining() > 10.0:
        h2, val2 = _local_polish(best_h)
        if val2 < best_val:
            best_val = val2
            best_h = h2.copy()
            
    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h, target_sum)
    
    return best_h

# EVOLVE-BLOCK-END
