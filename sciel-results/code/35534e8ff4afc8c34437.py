"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: exploit the analytic structure of the overlap functional.
The overlap C_k = 1 - (2/N) * r_k where r_k is the linear autocorrelation.
Minimizing max C_k is equivalent to maximizing min r_k.
Since r_k is a quadratic form, we can use a coordinate-descent / projected
gradient method on the log-sum-exp of -r_k (which is smooth and convex-ish
in the relevant region) with exact projection onto the polytope
{h in [0,1]^N, sum(h)=N/2}.
We also try a spectral initialization based on the eigenstructure of the
autocorrelation matrix.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 100
_DEADLINE = time.time() + 115


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """
    Project h onto the polytope: 0 <= h_i <= 1, sum(h) = N/2.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # If already close, return
    if abs(np.sum(h) - target) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        h = np.clip(h, 0.0, 1.0)
        # Fix tiny sum error
        diff = target - np.sum(h)
        if abs(diff) > 1e-14:
            # Distribute among non-boundary elements
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h
    
    # Bisection on lambda: h_i = clip(h_i + lambda, 0, 1)
    # We want sum(h_i) = target
    lo = -max(h) - 1.0
    hi = 1.0 - min(h) + 1.0
    
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            hi = mid
        else:
            lo = mid
    
    h_proj = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Final correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-14:
        mask = (h_proj > 1e-10) & (h_proj < 1 - 1e-10)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            # All at boundary, adjust one
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _lse_neg_r(h, beta):
    """
    Log-sum-exp of -r_k (negative autocorrelation values).
    Minimizing this is equivalent to maximizing min r_k.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    neg_r = -r
    m = np.max(neg_r)
    return m + np.log(np.sum(np.exp(beta * (neg_r - m)))) / beta


def _lse_neg_r_grad(h, beta):
    """
    Gradient of log-sum-exp(-r_k).
    d/dh_j of -r_k = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n) ... wait.
    Actually r_k = sum_{i=k}^{n-1} h[i] h[i-k]
    d r_k / d h_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k < n)
    So d(-r_k)/dh_j = -h_{j-k} - h_{j+k}
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    neg_r = -r
    m = np.max(neg_r)
    exp_terms = np.exp(beta * (neg_r - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # weights
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        # d(-r_k)/dh_j = -h_{j-k} (j>=k) - h_{j+k} (j+k<n)
        # So contribution to grad[j] from weight w[k] is:
        # w[k] * beta * [ -h_{j-k} - h_{j+k} ]
        # But we need to sum over k.
        # For a fixed j, the terms are:
        #   from k <= j: -h_{j-k}
        #   from k < n-j: -h_{j+k}
        # This is O(n^2) if done naively. Let's vectorize per k.
        
        # Term 1: for j >= k, add -w[k] * beta * h[j-k] to grad[j]
        if k > 0:
            grad[k:] -= beta * w[k] * h[:n-k]
        # Term 2: for j+k < n i.e. j < n-k, add -w[k] * beta * h[j+k] to grad[j]
        if k < n:
            grad[:n-k] -= beta * w[k] * h[k:]
            
    return val, grad


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Cosine
    i = np.arange(N)
    candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N))
    
    # 3. Sinusoidal with different frequencies
    for freq in [2, 3, 5]:
        candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N))
    
    # 4. Random
    for _ in range(5):
        candidates.append(np.random.uniform(0, 1, N))
    
    # 5. Spectral: try to make autocorrelation as flat as possible
    # The autocorrelation matrix A has entries A[i,j] = 1 if |i-j| < N
    # Actually r_k = sum_{i=k}^{n-1} h[i]h[i-k]
    # We want to maximize min_k r_k
    # This is a quadratic programming problem. Let's try a simple approach:
    # Initialize with the eigenvector of the "all-ones" structure.
    # Actually, let's just try a few more structured inits.
    candidates.append(0.5 + 0.3 * np.sin(2.0 * np.pi * i / N))
    candidates.append(0.5 + 0.3 * np.sin(4.0 * np.pi * i / N))
    
    # Project and evaluate all candidates
    for h in candidates:
        _update_best(h)
    
    # Gradient descent with projection
    # Use the smooth objective: log-sum-exp(-r_k)
    # Minimize this via projected gradient descent
    
    def _projected_grad_descent(h0, beta, steps, lr_init):
        nonlocal best_h, best_score
        h = h0.copy()
        lr = lr_init
        
        for step in range(steps):
            if time.time() > _DEADLINE - 5:
                break
            val, grad = _lse_neg_r_grad(h, beta)
            # Gradient step
            h_new = h - lr * grad
            # Project
            h_new = _project(h_new)
            
            # Adaptive learning rate: if objective increased, reduce lr
            val_new = _lse_neg_r(h_new, beta)
            if val_new > val + 1e-10:
                lr *= 0.5
                h_new = h - lr * grad
                h_new = _project(h_new)
            else:
                # Occasionally increase lr
                if step % 50 == 0 and lr < lr_init:
                    lr *= 1.1
                h = h_new
            
            # Update best
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        
        return h
    
    # Run projected gradient descent from the best candidate
    if best_h is not None:
        if time.time() < _DEADLINE - 30:
            _projected_grad_descent(best_h, beta=30.0, steps=500, lr_init=0.01)
        
        if time.time() < _DEADLINE - 20:
            _projected_grad_descent(best_h, beta=50.0, steps=500, lr_init=0.005)
        
        if time.time() < _DEADLINE - 10:
            _projected_grad_descent(best_h, beta=100.0, steps=300, lr_init=0.002)
    
    # Also try from other good candidates
    if best_h is not None:
        for h in candidates:
            if time.time() > _DEADLINE - 10:
                break
            h_proj = _project(h)
            s = _score(h_proj)
            if s < best_score + 0.05:  # Only if reasonably good
                _projected_grad_descent(h_proj, beta=50.0, steps=200, lr_init=0.005)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
