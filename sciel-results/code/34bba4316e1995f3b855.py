import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 130.0  # safety margin before 150s kill


def _convolve_fft(a, b):
    """Full convolution using FFT for speed."""
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()  # next power of 2
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    return np.fft.irfft(fa * fb, size)[:n]


def _soft_max(c, beta):
    """Smooth approximation to max(c) using log-sum-exp."""
    if beta < 1e-6:
        return float(np.max(c))
    m = float(np.max(c))
    shifted = (c - m) * beta
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / beta


def _soft_max_grad(c, beta):
    """Gradient of soft-max with respect to c."""
    if beta < 1e-6:
        idx = np.argmax(c)
        g = np.zeros_like(c)
        g[idx] = 1.0
        return g
    m = float(np.max(c))
    shifted = (c - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted)
    return probs


def _eval_value(h):
    """Evaluate the exact objective value."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf
    c = _convolve_fft(h, h)
    m = float(np.max(c))
    return 2.0 * N * m / (S * S)


def _soft_obj_and_grad(h, beta):
    """
    Soft objective: use soft-max instead of hard max.
    Returns (value, gradient).
    """
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf, np.zeros_like(h)
    
    c = _convolve_fft(h, h)
    m_soft = _soft_max(c, beta)
    val = 2.0 * N * m_soft / (S * S)
    
    # Gradient of soft_max w.r.t. c
    dc = _soft_max_grad(c, beta)
    
    # dc/dh[i] = sum_j dc[j] * dh[j]/dh[i]
    # c[j] = sum_i h[i]*h[j-i], so dc/dh[i] = sum_j dc[j] * h[j-i] (when 0<=j-i<N)
    # This is a convolution-like operation
    # dc/dh[i] = sum_{j: 0<=j-i<N} dc[j] * h[j-i]
    # Let k = j-i, so j = i+k, k in [0, N-1]
    # dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k] (when i+k < len(c))
    
    # More precisely: c[j] = sum_{i=0}^{N-1} h[i]*h[j-i] where j-i in [0,N-1]
    # dc/dh[i] = sum_j dc[j] * h[j-i] for j-i in [0, N-1]
    # = sum_{k=0}^{N-1} dc[i+k] * h[k] where i+k < 2N-1
    
    # Use FFT for this too
    # dc/dh[i] = (dc_conv)[i] where dc_conv = convolve(dc, h) but only valid part
    # Actually: dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k]
    # This is cross-correlation of dc and h
    
    # Let's compute it directly using the structure
    # c has length 2N-1, indices 0 to 2N-2
    # dc/dh[i] = sum_{j=max(i,0)}^{min(i+N-1, 2N-2)} dc[j] * h[j-i]
    
    # Use FFT: dc/dh = ifft(fft(dc, 2N-1) * conj(fft(h, 2N-1)))... 
    # Actually simpler: dc/dh[i] = sum_k dc[i+k]*h[k] = (dc * h)[i] in some sense
    # Let's just compute it via matrix-free approach with FFT
    
    # dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k] for i in [0, N-1]
    # This is like a sliding dot product
    
    # Pad dc to length 2N-1 (it already is)
    # dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k]
    # = cross-correlation
    
    # Use FFT: create vector v where v[k] = h[k] for k in [0,N-1], 0 otherwise
    # Then dc/dh[i] = sum_k dc[i+k] * v[k] = (dc * v)[i] where * is correlation
    # = ifft(conj(fft(dc)) * fft(v))[i]... 
    
    # Simpler: just do it with a loop over the structure
    # Actually, let's use the fact that this is a convolution
    # dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k]
    # Let's define w[k] = dc[N-1+k] for k in [0, N-1]... no that's not right
    
    # Direct computation using FFT:
    # dc/dh[i] = sum_{k=0}^{N-1} dc[i+k] * h[k]
    # = sum_{k=0}^{N-1} h[k] * dc[i+k]
    # This is the i-th element of the cross-correlation of h and dc
    
    # Cross-correlation: (h ⋆ dc)[i] = sum_k h[k] * dc[i+k]
    # = ifft(fft(h) * conj(fft(dc)))... 
    
    # Let me just compute it directly since N=400 is small
    # But we can use FFT:
    # Pad h to length 2N-1
    # dc/dh[i] = sum_{k=0}^{N-1} h[k] * dc[i+k]
    # This equals the i-th element of: ifft(fft(h_padded) * conj(fft(dc_padded)))
    # where h_padded and dc_padded are padded to length 2N-1
    
    size = 2 * N - 1
    h_pad = np.zeros(size)
    h_pad[:N] = h
    dc_pad = np.zeros(size)
    dc_pad[:size] = dc  # dc already has length 2N-1
    
    # Cross-correlation: sum_k h[k] * dc[i+k]
    # = ifft(conj(fft(h_pad)) * fft(dc_pad))[i]... 
    # Actually: (f ⋆ g)[i] = sum_k f[k] * g[i+k] = ifft(fft(f) * conj(fft(g)))
    # Wait, let me be careful.
    # Standard: ifft(fft(f) * conj(fft(g))) gives sum_k f[k] * g[k-i] (cross-correlation)
    # We want sum_k h[k] * dc[i+k] = sum_k h[k] * dc[i+k]
    # Let j = i+k, k = j-i: sum_j h[j-i] * dc[j] for j-i in [0,N-1]
    # = sum_j dc[j] * h[j-i]
    # This is (dc * h)[i] where * is convolution... 
    
    # Convolution: (a*b)[i] = sum_j a[j]*b[i-j]
    # We want sum_j dc[j] * h[i-j]... no, sum_j dc[j] * h[j-i]
    # = sum_j dc[j] * h[j-i] = (dc * h_rev)[i] where h_rev[k] = h[-k]... 
    
    # Let me just do it with direct computation for clarity, N=400 is fine
    # Actually, let's use the FFT approach properly:
    # We want g[i] = sum_{k=0}^{N-1} dc[i+k] * h[k] for i in [0, N-1]
    # This is the first N elements of the cross-correlation of dc and h
    
    # Cross-correlation R_dc_h[i] = sum_k dc[k] * h[k-i]... 
    
    # I'll just compute it directly - N=400, so O(N^2) = 160K ops, fast enough
    g = np.zeros(N)
    for i in range(N):
        # g[i] = sum_{k=0}^{N-1} dc[i+k] * h[k]
        # i+k ranges from i to i+N-1, must be < 2N-1
        valid_k = np.arange(N)
        valid_k = valid_k[(i + valid_k) < size]
        g[i] = np.sum(dc[i + valid_k] * h[valid_k])
    
    # Now gradient of val = 2N * m_soft / S^2
    # dval/dh[i] = 2N * (dc/dh[i] / S^2 - 2*m_soft*h[i]/S^3)
    grad_val = 2.0 * N * (g / (S * S) - 2.0 * m_soft * h / (S ** 3))
    
    return val, grad_val


def _exact_obj_and_grad(h):
    """Exact objective with subgradient for max."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf, np.zeros_like(h)
    
    c = _convolve_fft(h, h)
    m = float(np.max(c))
    jstar = int(np.argmax(c))
    val = 2.0 * N * m / (S * S)
    
    # Subgradient: dc/dh[i] = 2*h[jstar-i] when 0 <= jstar-i < N
    g = np.zeros(N)
    idx = jstar - np.arange(N)
    valid = (idx >= 0) & (idx < N)
    g[valid] = 2.0 * h[idx[valid]]
    
    grad_val = 2.0 * N * (g / (S * S) - 2.0 * m * h / (S ** 3))
    
    return val, grad_val


def construct():
    t0 = time.time()
    deadline = _DEADLINE - 5.0  # safety margin
    
    best_h = np.ones(N, dtype=float)
    best_val = _eval_value(best_h)  # seed value = 2.0
    
    rng = np.random.default_rng(20240601)
    
    # Generate diverse starting points
    start_points = []
    start_points.append(np.ones(N))
    start_points.append(rng.standard_normal(N))
    start_points.append(np.cumsum(rng.standard_normal(N)) - np.mean(rng.standard_normal(N)))
    start_points.append(np.sin(2 * np.pi * np.arange(N) / N) + 1.0)
    start_points.append(np.cos(2 * np.pi * np.arange(N) / N) + 1.0)
    start_points.append(np.sin(4 * np.pi * np.arange(N) / N) + 1.0)
    start_points.append(np.cos(4 * np.pi * np.arange(N) / N) + 1.0)
    start_points.append(np.sin(8 * np.pi * np.arange(N) / N) + 1.0)
    start_points.append(np.cos(8 * np.pi * np.arange(N) / N) + 1.0)
    
    # Add some structured patterns
    i = np.arange(N)
    start_points.append(np.sin(2 * np.pi * 3 * i / N) + 0.5)
    start_points.append(np.cos(2 * np.pi * 5 * i / N) + 0.5)
    start_points.append(np.sin(2 * np.pi * 7 * i / N) + 0.5)
    
    # Random patterns with different scales
    for scale in [0.5, 1.0, 2.0]:
        start_points.append(rng.standard_normal(N) * scale + 1.0)
    
    # Ensure all start points have non-zero sum
    for sp in start_points:
        S = float(np.sum(sp))
        if abs(S) < 1e-6:
            sp = sp + 1.0
        start_points[start_points.index(sp)] = sp
    
    # Phase 1: Soft-max optimization with adaptive temperature
    # Start with large beta (smooth), anneal to small beta (sharp)
    betas = [1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    
    for beta in betas:
        if (time.time() - t0) > (deadline - 10.0):
            break
        
        for h0 in start_points:
            if (time.time() - t0) > (deadline - 10.0):
                break
            
            h0_copy = h0.copy()
            
            def obj_soft(x, b=beta):
                v, _ = _soft_obj_and_grad(x, b)
                return v
            
            def grad_soft(x, b=beta):
                _, g = _soft_obj_and_grad(x, b)
                return g
            
            try:
                res = minimize(obj_soft, h0_copy, jac=grad_soft,
                               method="L-BFGS-B",
                               options={"maxiter": 200, "maxfun": 500,
                                        "ftol": 1e-12, "gtol": 1e-10})
                
                if np.isfinite(res.fun):
                    exact_val = _eval_value(res.x)
                    if exact_val < best_val:
                        best_val = exact_val
                        best_h = res.x.copy()
            except Exception:
                pass
        
        # Add best solution as a new starting point
        start_points.append(best_h.copy())
    
    # Phase 2: Exact optimization from best soft solutions
    if (time.time() - t0) < (deadline - 5.0):
        def obj_exact(x):
            v, _ = _exact_obj_and_grad(x)
            return v
        
        def grad_exact(x):
            _, g = _exact_obj_and_grad(x)
            return g
        
        # Start from best and some diverse points
        exact_starts = [best_h.copy()]
        for sp in start_points[:5]:
            exact_starts.append(sp.copy())
        
        for h0 in exact_starts:
            if (time.time() - t0) > (deadline - 5.0):
                break
            
            if abs(float(np.sum(h0))) < 1e-6:
                h0 = h0 + 1.0
            
            try:
                res = minimize(obj_exact, h0, jac=grad_exact,
                               method="L-BFGS-B",
                               options={"maxiter": 300, "maxfun": 1000,
                                        "ftol": 1e-14, "gtol": 1e-12})
                
                if np.isfinite(res.fun):
                    exact_val = _eval_value(res.x)
                    if exact_val < best_val:
                        best_val = exact_val
                        best_h = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Final polish with very tight tolerances
    if (time.time() - t0) < (deadline - 3.0):
        def obj_exact(x):
            v, _ = _exact_obj_and_grad(x)
            return v
        
        def grad_exact(x):
            _, g = _exact_obj_and_grad(x)
            return g
        
        try:
            res = minimize(obj_exact, best_h, jac=grad_exact,
                           method="L-BFGS-B",
                           options={"maxiter": 500, "maxfun": 2000,
                                    "ftol": 1e-16, "gtol": 1e-14})
            
            if np.isfinite(res.fun):
                exact_val = _eval_value(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final: ensure validity
    S = float(np.sum(best_h))
    if abs(S) > 1e-9:
        best_h = best_h / S
    else:
        best_h = best_h + 1.0
    
    # Final verification
    final_val = _eval_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N, dtype=float)
    
    return best_h
