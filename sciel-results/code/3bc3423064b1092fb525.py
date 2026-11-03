import numpy as np
import time

def _autoconv(f):
    """Compute autoconvolution c_k = sum_i f_i * f_{i+k} for k in [-n+1, n-1].
    Returns array of length 2n-1 where index n-1 corresponds to k=0."""
    n = len(f)
    # Use FFT: autoconv = ifft(fft(f) * conj(fft(f)))
    F = np.fft.rfft(f, n=2*n)
    ac = np.fft.irfft(F * np.conj(F), n=2*n)[:n]
    # ac[i] for i=0..n-1 gives sum_j f[j] f[j+i] (circular)
    # We want linear autoconv: c_k = sum_j f[j] f[j+k] for k >= 0
    # The circular version wraps around, so we need to handle it
    # Actually for non-negative f, the linear autoconv at lag k is:
    # c_k = sum_{j=0}^{n-1-k} f[j] * f[j+k]
    # Let's just compute it directly for safety, or use the FFT trick properly
    # Better: zero-pad to 2n and compute
    F_full = np.fft.fft(f, n=2*n)
    ac_full = np.fft.ifft(F_full * np.conj(F_full))[:n].real
    # ac_full[k] = sum_{j=0}^{n-1-k} f[j] f[j+k] for k=0,...,n-1
    # For negative lags, ac_full[n-1-k] = sum f[j+k] f[j] = same by symmetry
    # So the full autoconv (lags -(n-1) to n-1) is:
    # result[i] for i=0..2n-2 corresponds to lag i-(n-1)
    # result[n-1] = c_0 = sum f[j]^2
    # result[n-1+k] = c_k = sum f[j]f[j+k] for k=1..n-1
    # result[n-1-k] = c_{-k} = c_k (symmetric for real)
    full_ac = np.zeros(2*n-1)
    full_ac[n-1:] = ac_full  # lags 0 to n-1
    full_ac[:n-1] = ac_full[1:][::-1]  # lags -(n-1) to -1
    return full_ac

def _compute_score(f):
    """Compute the normalized max of autoconvolution.
    The autoconvolution c_k = sum_i f_i f_{i+k}.
    Normalized: max_k |c_k| / (sum f_i)^2 * (1/interval_width) or similar.
    Actually, for step functions on equal intervals, the autoconvolution integral is:
    (c_k) * dx where dx = interval width.
    The normalized quantity is max_k (c_k * dx) / (sum f_i * dx)^2 * ... 
    Let me think about this more carefully.
    
    For a step function f(x) = f_i on interval [i*dx, (i+1)*dx), the autoconvolution is:
    (f*f)(t) = integral f(x) f(t-x) dx
    For t = k*dx (integer multiple), this is approximately sum_i f_i f_{i+k} * dx
    
    The "normalized maximum" likely means max_t (f*f)(t) / (integral f)^2 * something
    or max_k c_k / c_0 where c_0 = sum f_i^2
    
    Given the task says "normalized maximum of its autoconvolution" and "upper bound on C1",
    and the seed returns [1.0]*600 with score 1.0, let's figure out the normalization.
    
    For f = [1]*600: c_0 = 600, c_k = 600-|k| for |k|<600
    max c_k = c_0 = 600
    If score = max_k c_k / c_0 = 1.0, that matches!
    Or score = max_k c_k / (sum f_i)^2 * n = 600 / 600^2 * 600 = 1.0
    Or score = max_k c_k / sum(f_i * f_i) = 600/600 = 1.0
    
    Let's assume score = max_k c_k / c_0 = max_k c_k / sum_i f_i^2
    """
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0  # degenerate
    c0 = np.dot(f, f)
    if c0 == 0:
        return 0.0
    ac = _autoconv(f)
    max_ac = np.max(ac)
    return max_ac / c0

def _subgradient_step(f, ac, lr):
    """One subgradient descent step to minimize max_k ac_k / c0."""
    n = len(f)
    c0 = np.dot(f, f)
    # The objective is J(f) = max_k ac_k / c0
    # ac_k = sum_i f_i f_{i+k} (with appropriate bounds)
    # dJ/df_i involves which k achieves the max
    # Let's use a simpler approach: soft max
    # J ~ log(sum_k exp(beta * ac_k)) / c0
    beta = 50.0
    ac_shifted = ac - np.max(ac)
    soft_max = np.sum(np.exp(beta * ac_shifted)) / (beta * len(ac))
    # Gradient of soft_max w.r.t. f_i:
    # d(soft_max)/df_i = (1/c0) * sum_k p_k * dac_k/df_i where p_k = exp(beta*ac_k)/sum
    # dac_k/df_i = f_{i-k} + f_{i+k} (for valid indices)
    p = np.exp(beta * ac_shifted)
    p /= np.sum(p)
    # Compute gradient of soft_max w.r.t. f
    grad_sm = np.zeros(n)
    for k_idx in range(len(ac)):
        k = k_idx - (n-1)  # lag
        pk = p[k_idx]
        if pk < 1e-10:
            continue
        # dac_k/df_i = f_{i-k} + f_{i+k} for valid i
        # For lag k >= 0: ac_k = sum_{i=0}^{n-1-k} f_i f_{i+k}
        # d(ac_k)/df_i = f_{i+k} (if i+k < n) + f_{i-k} (if i-k >= 0)
        # For simplicity, use FFT-based gradient
        pass
    # Use FFT for efficiency
    F = np.fft.fft(f, n=2*n)
    # ac_full from FFT: ac_full[k] for k=0..n-1 is sum f_i f_{i+k}
    # The gradient of sum_k p_k ac_k w.r.t. f is:
    # For each k, d(ac_k)/df = ifft(conj(fft(f shifted by k))) + ifft(conj(fft(f shifted by -k)))
    # This is getting complex. Let's use a direct approach for 600 elements.
    
    # Direct computation of gradient
    grad = np.zeros(n)
    for k_idx in range(len(ac)):
        k = k_idx - (n-1)
        pk = p[k_idx]
        if pk < 1e-10:
            continue
        if k >= 0:
            # ac_k = sum_{i=0}^{n-1-k} f[i] * f[i+k]
            # d(ac_k)/df[i] = f[i+k] for i=0..n-1-k, and f[i-k] for i=k..n-1
            # Combined: d(ac_k)/df[i] = f[i+k] (if i+k<n) + f[i-k] (if i-k>=0)
            for i in range(n):
                if i + k < n:
                    grad[i] += pk * f[i+k]
                if i - k >= 0:
                    grad[i] += pk * f[i-k]
        else:
            # k < 0, let k' = -k > 0
            # ac_{-k'} = sum_{i=k'}^{n-1} f[i] * f[i-k'] = sum_{j=0}^{n-1-k'} f[j+k'] * f[j]
            # Same as ac_{k'} by symmetry
            kp = -k
            for i in range(n):
                if i + kp < n:
                    grad[i] += pk * f[i+kp]
                if i - kp >= 0:
                    grad[i] += pk * f[i-kp]
    
    # J = soft_max / c0, so dJ/df = (grad * c0 - soft_max * 2*f) / c0^2
    # = grad/c0 - 2*f*soft_max/c0^2
    # = (grad - 2*f*soft_max/c0) / c0
    obj_val = soft_max / c0
    grad_obj = (grad - 2.0 * f * soft_max / c0) / c0
    
    # Project to non-negative
    f_new = f - lr * grad_obj
    f_new = np.maximum(f_new, 0)
    return f_new

def _coordinate_descent(f, max_iter=200, tol=1e-10):
    """Coordinate descent to polish the solution."""
    n = len(f)
    f = np.array(f, dtype=np.float64, copy=True)
    best_score = _compute_score(f)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Try different values for f[i]
            old_val = f[i]
            # Optimal f[i] given others: minimize max_k ac_k / c0
            # ac_k depends on f[i] quadratically
            # For simplicity, try a few candidate values
            # The optimal f[i] is where the derivative of the objective w.r.t. f[i] is zero
            # This is complex, so let's just try shrinking or growing
            if old_val > 1e-12:
                # Try reducing
                for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                    f[i] = old_val * factor
                    s = _compute_score(f)
                    if s < best_score - tol:
                        best_score = s
                        improved = True
                        break
            # Try setting to 0
            f[i] = 0.0
            s = _compute_score(f)
            if s < best_score - tol:
                best_score = s
                improved = True
                break
            # Restore
            f[i] = old_val
        if not improved:
            break
    return f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 600
    # Initialize with uniform
    f = np.ones(n, dtype=np.float64)
    best_f = f.copy()
    best_score = _compute_score(f)
    
    # Subgradient descent
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        ac = _autoconv(f)
        f = _subgradient_step(f, ac, lr)
        # Normalize: scaling f by constant doesn't change score (ratio)
        # But we need to keep values reasonable
        if np.any(f > 0):
            f = f / np.max(f)
        score = _compute_score(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.9
    
    # Coordinate descent polish
    if time.time() < deadline - 5.0:
        f_cd, score_cd = _coordinate_descent(best_f, max_iter=50)
        if score_cd < best_score:
            best_score = score_cd
            best_f = f_cd
    
    return list(best_f)
