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

def _square_wave_score(a):
    """Compute score for square wave: +1 for k<a, -1 for k>=a.
    a is the transition point (number of +1's)."""
    h = np.ones(N)
    h[a:] = -1.0
    s = 2.0 * a - N
    if abs(s) < 1e-12:
        return 1e15
    ac = _autoconv_full(h)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

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
        
        # Compute gradient via FFT
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        
        G = np.fft.rfft(grad_ac, fsize)
        H = np.fft.rfft(h, fsize)
        grad_h = np.fft.irfft(G * np.conj(H), fsize)[:n]
        grad_h = 2.0 * grad_h
        
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Phase 1: Grid search over square wave transition points
    # Square wave: +1 for k < a, -1 for k >= a
    # The score is scale-invariant, so we just need to find the best a
    for a in range(1, N):
        if time.time() > deadline - 60:
            break
        val = _square_wave_score(a)
        if val < best_val:
            best_val = val
            best_h = np.ones(N)
            best_h[a:] = -1.0
    
    # Phase 2: Also try reversed square wave: -1 for k < a, +1 for k >= a
    for a in range(1, N):
        if time.time() > deadline - 40:
            break
        h = np.ones(N)
        h[:a] = -1.0
        s = 2.0 * (N - a) - N
        if abs(s) < 1e-12:
            continue
        ac = _autoconv_full(h)
        max_abs = np.max(np.abs(ac))
        val = max_abs / (s * s)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 3: Gradient descent polish from best square wave
    if time.time() < deadline - 10:
        h_opt, val_opt = _gradient_descent(best_h, deadline - 5, lr=0.01, beta=50.0, max_iters=2000)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 4: Final fine polish
    if time.time() < deadline - 3:
        h_final, val_final = _gradient_descent(best_h, deadline - 1, lr=0.003, beta=100.0, max_iters=3000)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
