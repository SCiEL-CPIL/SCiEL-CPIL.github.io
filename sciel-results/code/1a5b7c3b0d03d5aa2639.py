"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines linear overlap definition (FFT-accelerated) with analytical gradient optimization.
Structural change: fully vectorized gradient via FFT (O(N log N) per gradient eval),
epigraph-free smooth-max SLSQP, multiple structured starts, and multi-phase polishing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective using fully vectorized FFT-based approach.
    Returns (value, gradient).
    
    Key insight: dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    The gradient can be computed as a weighted sum of shifted h vectors.
    We use FFT to compute this in O(N log N) instead of O(N^2).
    
    Specifically, grad_j = -(2/n) * sum_k w_k * (h_{j-k} * 1[j>=k] + h_{j+k} * 1[j+k<n])
    This is equivalent to: grad = -(2/n) * (w * h)_linear_corr
    where the correlation is computed with appropriate boundary handling.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Fully vectorized gradient via FFT
    # grad_j = -(2/n) * sum_k w_k * h_{j-k} (for j>=k) + w_k * h_{j+k} (for j+k<n)
    # This is: grad = -(2/n) * (linear_cross_correlation of w and h, appropriately defined)
    
    # Let's define it more carefully:
    # Term A: sum_k w_k * h_{j-k} for j >= k, i.e., k <= j
    #   = sum_{k=0}^{min(j,n-1)} w_k * h_{j-k}
    #   This is the linear cross-correlation (w * h)[j] = sum_k w_k * h_{j-k}
    # 
    # Term B: sum_k w_k * h_{j+k} for j+k < n, i.e., k < n-j
    #   = sum_{k=0}^{n-j-1} w_k * h_{j+k}
    #   This is the linear cross-correlation (h * w)[j] = sum_k h_{j+k} * w_k
    #
    # So grad = -(2/n) * (TermA + TermB)
    #
    # We can compute both via FFT:
    # TermA[j] = sum_{k=0}^{j} w[k] * h[j-k] for j=0..n-1  (linear cross-corr of w and h)
    # TermB[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k] for j=0..n-1  (linear cross-corr of h and w, flipped)
    
    n_fft = 2 * n
    
    # Compute TermA: linear cross-correlation w * h
    # (w * h)[j] = sum_k w[k] * h[j-k] where valid indices are 0<=k<n, 0<=j-k<n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    # Cross-correlation: ifft(conj(W) * H) gives sum_k w[k] * h[j+k]
    # We want sum_k w[k] * h[j-k], which is ifft(W * conj(H))
    termA_full = np.fft.irfft(W * np.conj(H), n=n_fft)
    termA = termA_full[:n]
    
    # Compute TermB: sum_k w[k] * h[j+k] for k=0..n-1-j
    # This is ifft(conj(W) * H)
    termB_full = np.fft.irfft(np.conj(W) * H, n=n_fft)
    termB = termB_full[:n]
    
    grad = -(2.0 / n) * (termA + termB)
    
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    i = np.arange(N)
    
    # Phase 1: Cosine start (primary)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    if time.time() - t0 < _DEADLINE - 35:
        _run_optimization(h_cos, beta=50.0, maxiter=300, ftol=1e-9)
    
    # Phase 2: Sine start
    if time.time() - t0 < _DEADLINE - 30:
        h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
        h_sin = _fix_sum(h_sin)
        _update_best(h_sin)
        _run_optimization(h_sin, beta=50.0, maxiter=200, ftol=1e-9)
    
    # Phase 3: Triangle wave
    if time.time() - t0 < _DEADLINE - 25:
        h_tri = 0.5 + 0.5 * np.abs(2.0 * ((i / N) % 1.0) - 1.0)
        h_tri = _fix_sum(h_tri)
        _update_best(h_tri)
        _run_optimization(h_tri, beta=50.0, maxiter=200, ftol=1e-9)
    
    # Phase 4: Random starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 12:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=150, ftol=1e-8)
    
    # Phase 5: Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        _run_optimization(best_h, beta=100.0, maxiter=400, ftol=1e-12)
    
    # Phase 6: Ultra-fine polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-13)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
