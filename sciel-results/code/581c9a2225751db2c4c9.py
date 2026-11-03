"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_normalized(h):
    """
    Compute the objective: max|autoconv| / (sum h)^2.
    We assume h is normalized to sum=1, so this is just max|autoconv|.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    max_ac = np.max(np.abs(ac[:size]))
    return max_ac / (s * s)

def _gradient_descent_continuous(h_init, steps=2000, eta=0.01, deadline=None):
    """
    Minimizes max|autoconv| using subgradient descent on continuous values.
    """
    h = h_init.copy().astype(float)
    n = len(h)
    
    # Adam optimizer state
    m = np.zeros(n)
    v = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Normalize h to sum 1
        s = np.sum(h)
        if abs(s) < 1e-9:
            h = h_init.copy()
            s = 1.0
        else:
            h = h / s
        
        # Compute autoconvolution
        H = np.fft.rfft(h, 2*n)
        ac = np.fft.irfft(H * H, 2*n)
        ac = ac[:2*n-1]
        
        # Find peaks
        max_ac = np.max(np.abs(ac))
        if max_ac < 1e-6:
            break
        
        # Identify indices of peaks (top few)
        # We use a threshold to include all significant peaks
        threshold = 0.9 * max_ac
        peak_indices = np.where(np.abs(ac) >= threshold)[0]
        
        # Compute subgradient
        # grad_k[i] = 2 * h[(k-i) % n]
        # We can compute this efficiently using FFT or direct rolls for few peaks
        # Since N=400, direct rolls for ~10-20 peaks is fast
        g = np.zeros(n)
        
        # Precompute flipped h for faster rolls?
        # h_flip[i] = h[(-i) % n]
        h_flip = np.zeros(n)
        h_flip[0] = h[0]
        h_flip[1:] = h[:0:-1]
        
        for k in peak_indices:
            # sign(c_k)
            sign_k = np.sign(ac[k])
            # grad contribution: 2 * sign_k * roll(h_flip, k)
            # roll(h_flip, k)[i] = h_flip[i-k] = h[-(i-k)] = h[k-i]
            # This matches h[(k-i) % n]
            g += sign_k * 2.0 * np.roll(h_flip, k)
        
        # Update Adam state
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        
        # Update h
        h -= eta * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project to sum 1 (subtract mean)
        h -= np.mean(h)
        
    # Final normalization
    s = np.sum(h)
    if abs(s) > 1e-9:
        h = h / s
    else:
        h = h_init / np.sum(h_init)
        
    return h

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_val = np.inf
    best_h = np.ones(N) / N
    
    best_val = _score_normalized(best_h)
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Continuous Gradient Descent from Chirp Initialization
    # Chirps have good auto-correlation properties
    x = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        phase = 2 * np.pi * freq * x**2 / (2 * N)
        h_init = np.cos(phase)
        # Ensure sum is not zero
        if abs(np.sum(h_init)) < 1e-6:
            h_init += 1.0
            h_init /= np.sum(h_init)
        else:
            h_init /= np.sum(h_init)
            
        h_opt = _gradient_descent_continuous(h_init, steps=500, eta=0.1, deadline=deadline - 5)
        val = _score_normalized(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
        
        if time.time() > deadline - 10:
            break
            
    # Strategy 2: Continuous Gradient Descent from Random Initialization
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        h_init = rng.normal(size=N)
        h_init /= np.sum(h_init)
        
        h_opt = _gradient_descent_continuous(h_init, steps=500, eta=0.1, deadline=deadline - 5)
        val = _score_normalized(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt

    return best_h

# EVOLVE-BLOCK-END
