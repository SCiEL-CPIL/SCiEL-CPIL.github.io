import numpy as np
import time

N = 400

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c_excl, beta=50.0):
    abs_c = np.abs(c_excl)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _objective_offdiag(x, beta=50.0):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    return _soft_max_abs(c[1:], beta) / (s * s)

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Strategy: Use signed sequences with iterative peak cancellation
    # Start from multiple different initializations
    
    # Init 1: Hanning (positive)
    # Init 2: Hanning * sign(chirp)
    # Init 3: Random signed, normalized
    # Init 4: Alternating with smoothing
    
    inits = []
    
    # Init 1: Hanning
    inits.append(np.hanning(N))
    
    # Init 2: Hanning modulated by sign of cosine
    t = np.arange(N)
    inits.append(np.hanning(N) * np.sign(np.cos(2 * np.pi * t / N)))
    
    # Init 3: Random signed, then make sum positive
    x_rand = rng.choice([-1, 1], size=N).astype(float)
    if np.sum(x_rand) < 0:
        x_rand = -x_rand
    inits.append(x_rand)
    
    # Init 4: Chirp-like signed
    inits.append(np.sign(np.cos(2 * np.pi * t * t / N)))
    
    # Init 5: Hanning with alternating sign blocks
    block = 8
    signs = np.ones(N)
    for i in range(0, N, 2 * block):
        signs[i:i+block] = -1
    inits.append(np.hanning(N) * signs)
    
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Normalize sum to be positive and reasonable
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            # Add small constant to make sum positive
            x0 = x0 + 0.1
            s0 = np.sum(x0)
        x0 = x0 / s0 * N  # Normalize so sum = N
        
        # Iterative peak cancellation
        x = x0.copy()
        for it in range(200):
            if time.time() > deadline:
                break
            
            c = _autoconv_lin(x)
            s = np.sum(x)
            if abs(s) < 1e-6:
                break
            
            # Find worst off-diagonal lag
            if len(c) <= 1:
                break
            abs_c = np.abs(c[1:])
            k_star = np.argmax(abs_c) + 1  # actual lag index
            
            # Gradient of c[k_star] w.r.t. x[j]:
            # c[k] = sum_i x[i] x[i-k]
            # dc[k]/dx[j] = x[j-k] + x[j+k] (with zero padding)
            grad = np.zeros(N)
            # x[j-k] for j >= k: x[j-k], for j < k: 0
            if k_star < N:
                grad[k_star:] = x[:N-k_star]
                # x[j+k] for j+k < N: x[j+k], for j+k >= N: 0
                grad[:N-k_star] += x[k_star:]
            
            # We want to reduce |c[k_star]|, so move x in direction -sign(c[k_star]) * grad
            step = 0.01
            x = x - step * np.sign(c[k_star]) * grad
            
            # Renormalize sum to N
            s_new = np.sum(x)
            if abs(s_new) < 1e-6:
                x = x0.copy()
                break
            x = x / s_new * N
        
        # Evaluate
        val = _metric_offdiag(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # If no valid result, fallback to Hanning
    if best_x is None:
        best_x = np.hanning(N)
        best_val = _metric_offdiag(best_x)
    
    return best_x
