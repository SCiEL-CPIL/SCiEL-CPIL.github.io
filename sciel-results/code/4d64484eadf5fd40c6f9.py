"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_full(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_full(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e15
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Soft approximation of max|ac| using log-sum-exp for smooth optimization."""
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-15:
        return 0.0
    shifted = beta * (x - m)
    shifted = np.clip(shifted, -50, 50)
    s = np.sum(np.exp(shifted))
    return m + (1.0 / beta) * np.log(s)

def _smooth_score(h, beta=50.0):
    """Smooth objective for gradient-based optimization."""
    ac = _autoconv_full(h)
    smax = _soft_max_abs(ac, beta)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e15
    return smax / (s * s)

def _gradient_descent(h0, deadline, lr=0.01, beta=50.0, max_iters=5000):
    """Gradient descent on continuous heights using smooth surrogate."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _score(h)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute smooth objective and gradient
        ac = _autoconv_full(h)
        x = np.abs(ac)
        m = np.max(x)
        if m > 1e-15:
            shifted = beta * (x - m)
            shifted = np.clip(shifted, -50, 50)
            exp_shifted = np.exp(shifted)
            w = exp_shifted / np.sum(exp_shifted)
            # Gradient of soft_max_abs wrt ac
            grad_ac = beta * w * np.sign(ac)
        else:
            grad_ac = np.zeros_like(ac)
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            break
        
        # d(score)/d(h) = d(smax)/d(h) / s^2 - 2*smax/s^3 * d(s)/d(h)
        # d(smax)/d(h)_i = sum_j grad_ac[j] * d(ac_j)/d(h_i)
        # ac_j = sum_k h_k * h_{j-k}, so d(ac_j)/d(h_i) = h_{j-i} + h_{i-j} (if in range)
        # Using FFT: d(ac)/d(h_i) is the convolution of delta_i with h
        # More efficiently: grad_h[i] = sum_j grad_ac[j] * (h[j-i] + h[i-j])
        # This is a correlation-like operation
        
        # Compute gradient via FFT
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        
        # grad_h[i] = sum_j grad_ac[j] * h[j-i] + sum_j grad_ac[j] * h[i-j]
        # First term: (grad_ac * h)[i] where * is cross-correlation
        # Use FFT: IFFT(FFT(grad_ac) * conj(FFT(h)))
        G = np.fft.rfft(grad_ac, fsize)
        H = np.fft.rfft(h, fsize)
        grad_h = np.fft.irfft(G * np.conj(H), fsize)[:n]
        
        # Second term is the same by symmetry of autoconvolution
        # Actually for autoconvolution: ac[j] = sum_k h[k]*h[j-k]
        # d(ac[j])/d(h[i]) = h[j-i] + h[i-j] (when indices valid)
        # So grad_h[i] = sum_j grad_ac[j] * (h[j-i] + h[i-j])
        # The second sum is just the first with j and i swapped, which equals the same thing
        # Wait, let me reconsider. 
        # ac[j] = sum_k h[k] h[j-k]
        # d(ac[j])/d(h[i]) = h[j-i] (from k=i) + h[i-j] (from j-k=i, i.e., k=j-i)
        # So grad_h[i] = sum_j grad_ac[j] h[j-i] + sum_j grad_ac[j] h[i-j]
        # Let m = j-i in first sum: sum_m grad_ac[m+i] h[m]
        # Let m = i-j in second sum: sum_m grad_ac[i-m] h[m]
        # These are cross-correlations
        
        # Using FFT for cross-correlation:
        # First part: corr1[i] = sum_m grad_ac[m+i] h[m] = (grad_ac ⊛ h)[i]
        # Second part: corr2[i] = sum_m grad_ac[i-m] h[m] = (grad_ac * h)[i] (convolution)
        
        # Actually simpler: since autoconv is symmetric, the gradient is:
        # grad_h = 2 * IFFT(FFT(grad_ac) * conj(FFT(h)))[:n]
        # Let me verify with a small example mentally...
        # For now, use the FFT approach
        grad_h = 2.0 * grad_h  # factor of 2 from both terms
        
        # Now total gradient of score = smax/s^2
        # d(score)/d(h_i) = (1/s^2) * grad_h[i] - 2*smax/s^3 * 1
        smax = _soft_max_abs(ac, beta)
        grad_score = grad_h / (s * s) - 2.0 * smax / (s * s * s)
        
        # Update
        h = h - lr * grad_score
        
        # Check exact score
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr = max(0.001, lr * 0.95)
    
    return best_h, best_val

def _flat_top_window(N, alpha=0.5):
    """Generate a flat-top window: 1 in center, tapered at edges."""
    x = np.arange(N)
    mid = N / 2.0
    # Raised cosine taper in the outer alpha fraction
    taper_start = int(alpha * N)
    h = np.ones(N)
    # Left taper
    for i in range(taper_start):
        t = i / taper_start
        h[i] = 0.5 * (1 + np.cos(np.pi * t))
    # Right taper
    for i in range(taper_start):
        t = (N - 1 - i) / taper_start
        h[N - 1 - i] = 0.5 * (1 + np.cos(np.pi * t))
    return h

def _hanning_window(N):
    """Hanning window."""
    x = np.arange(N)
    return 0.5 * (1 + np.cos(2 * np.pi * x / (N - 1)))

def _blackman_window(N):
    """Blackman window."""
    x = np.arange(N)
    a0, a1, a2 = 0.42, 0.5, 0.08
    return a0 - a1 * np.cos(2 * np.pi * x / (N - 1)) + a2 * np.cos(4 * np.pi * x / (N - 1))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Phase 1: Generate diverse window initializations
    initializations = []
    
    # Flat-top windows with different alpha
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        h = _flat_top_window(N, alpha)
        initializations.append(h)
    
    # Hanning
    initializations.append(_hanning_window(N))
    
    # Blackman
    initializations.append(_blackman_window(N))
    
    # Triangular
    x = np.arange(N)
    mid = N / 2.0
    tri = 1.0 - np.abs(x - mid) / mid
    initializations.append(tri)
    
    # Gaussian-like
    x = np.arange(N)
    mid = (N - 1) / 2.0
    for sigma in [N*0.1, N*0.15, N*0.2, N*0.3]:
        gauss = np.exp(-0.5 * ((x - mid) / sigma) ** 2)
        initializations.append(gauss)
    
    # Cosine taper (raised cosine)
    for taper_frac in [0.1, 0.2, 0.3, 0.4]:
        h = np.ones(N)
        tlen = int(taper_frac * N)
        for i in range(tlen):
            t = i / tlen
            h[i] = 0.5 * (1 + np.cos(np.pi * t))
            h[N-1-i] = 0.5 * (1 + np.cos(np.pi * t))
        initializations.append(h)
    
    # Evaluate all initializations
    candidates = []
    for h in initializations:
        val = _score(h)
        candidates.append((val, h))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Sort and take top 15
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:15]
    
    # Phase 2: Gradient descent on top candidates
    for val_init, h_init in top_candidates:
        if time.time() > deadline - 10:
            break
        
        h_opt, val_opt = _gradient_descent(h_init, deadline - 5, lr=0.02, beta=50.0, max_iters=3000)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Final polish with finer gradient descent
    if time.time() < deadline - 5:
        h_final, val_final = _gradient_descent(best_h, deadline - 1, lr=0.005, beta=100.0, max_iters=5000)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
