import numpy as np
import time

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

def _subgradient_step(h, lr, n_active_thresh=1e-6):
    """Take a subgradient step to reduce max overlap."""
    n = len(h)
    corrs = _compute_overlaps(h)
    max_val = np.max(corrs)
    
    if max_val < 1e-15:
        return h, max_val
    
    # Find active constraints (near-max overlaps)
    threshold = max_val - n_active_thresh
    active = corrs >= threshold
    
    subgrad = np.zeros(n)
    for k in range(1, n):
        if not active[k-1]:
            continue
        # R_k = (2/N) * sum_{i=0}^{n-k-1} h[i] * h[i+k]
        # dR_k/dh[j] = (2/N) * (h[j+k] if j < n-k else 0) + (2/N) * (h[j-k] if j >= k else 0)
        w = corrs[k-1] / max_val
        for j in range(n - k):
            subgrad[j] += w * h[j + k]
            subgrad[j + k] += w * h[j]
        subgrad *= (2.0 / n)
    
    # Normalize
    norm = np.linalg.norm(subgrad)
    if norm > 1e-15:
        subgrad /= norm
    
    h_new = h - lr * subgrad
    h_new = _project_to_feasible(h_new, N / 2.0)
    return h_new, max_val

def _fourier_optimize(h, max_iters=2000, lr_init=0.1):
    """Optimize in Fourier space using subgradient method."""
    n = len(h)
    target_sum = n / 2.0
    
    # Convert to Fourier coefficients (centered)
    h_centered = h - np.mean(h)
    coeffs = np.fft.fft(h_centered) / n
    
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    lr = lr_init
    patience = 0
    best_iter_val = best_val
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Convert back to time domain
        h_cur = np.real(np.fft.ifft(coeffs * n)) + 0.5
        h_cur = _project_to_feasible(h_cur, target_sum)
        
        # Compute subgradient in time domain
        corrs = _compute_overlaps(h_cur)
        max_val = np.max(corrs)
        
        if max_val < best_iter_val - 1e-10:
            best_iter_val = max_val
            patience = 0
        else:
            patience += 1
        
        if patience > 100:
            lr *= 0.9
            patience = 0
            if lr < 1e-6:
                break
        
        # Find active shifts
        threshold = max_val - 1e-6
        active = corrs >= threshold
        
        # Compute subgradient in time domain
        subgrad = np.zeros(n)
        for k in range(1, n):
            if not active[k-1]:
                continue
            w = corrs[k-1] / max_val if max_val > 1e-15 else 1.0
            w *= (2.0 / n)
            for j in range(n - k):
                subgrad[j] += w * h_cur[j + k]
                subgrad[j + k] += w * h_cur[j]
        
        # Convert subgradient to Fourier domain
        subgrad_centered = subgrad - np.mean(subgrad)
        subgrad_coeffs = np.fft.fft(subgrad_centered) / n
        
        # Update Fourier coefficients
        coeffs -= lr * subgrad_coeffs
        
        # Track best
        h_test = np.real(np.fft.ifft(coeffs * n)) + 0.5
        h_test = _project_to_feasible(h_test, target_sum)
        val_test = _max_overlap_value(h_test)
        if val_test < best_val:
            best_val = val_test
            best_h = h_test.copy()
    
    return best_h, best_val

def _soft_min_polish(h, temps=[0.1, 0.05, 0.02, 0.01]):
    """Polish using soft-min (log-sum-exp) approximation."""
    from scipy.optimize import minimize
    
    n = len(h)
    
    def _obj_and_grad(h_vec, temp):
        h_vec = np.clip(h_vec, 0, 1)
        corrs = _compute_overlaps(h_vec)
        shifted = corrs / temp
        max_sh = np.max(shifted)
        logsumexp = max_sh + np.log(np.sum(np.exp(shifted - max_sh)))
        obj = temp * logsumexp
        
        # Gradient via finite differences or analytical
        # Use analytical: d/dh of softmax(corrs/temp) * dcorrs/dh
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
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 20, 'maxfun': 500}
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

def construct():
    t0 = time.time()
    target_sum = N / 2.0
    
    # Start with spectral initialization
    h = _spectral_init()
    best_h = h.copy()
    best_val = _max_overlap_value(h)
    
    # Phase 1: Fourier-space subgradient optimization
    if _time_remaining() > 20.0:
        h1, val1 = _fourier_optimize(h, max_iters=3000, lr_init=0.05)
        if val1 < best_val:
            best_val = val1
            best_h = h1.copy()
    
    # Phase 2: Soft-min polishing
    if _time_remaining() > 15.0:
        h2, val2 = _soft_min_polish(best_h, temps=[0.2, 0.1, 0.05, 0.02, 0.01])
        if val2 < best_val:
            best_val = val2
            best_h = h2.copy()
    
    # Final projection to ensure feasibility
    best_h = _project_to_feasible(best_h, target_sum)
    
    return best_h

# EVOLVE-BLOCK-END
