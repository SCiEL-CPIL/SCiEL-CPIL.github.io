"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time


def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]


def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Strategy:
    # 1. Generate a high-quality initial candidate using a "balanced" signed sequence.
    #    The Thue-Morse sequence has the property that its autocorrelation is relatively
    #    flat compared to random sequences, which helps minimize the max absolute value
    #    relative to the integral (especially if we map to real values or keep as +/-1 
    #    with a DC offset to make sum non-zero).
    #    
    #    However, the task asks for "signed step heights" and the score is 
    #    max(|autoconv|) / (sum(x))^2.
    #    
    #    If we use pure +/-1, sum(x) is small (0 for Thue-Morse of length power of 2), 
    #    which makes the denominator tiny and score huge. 
    #    We need a large sum(x) to minimize the score. 
    #    Therefore, we should use values in a range like [0, 2] or [1, 3] or similar 
    #    to keep the mean high, but with fluctuations that reduce the peak autoconv.
    #    
    #    Let's try: x_i = 1 + a * t_i, where t_i is a low-discrepancy sequence in {-1, 1}.
    #    This keeps the mean at 1, so sum(x) ~ N.
    #    The autoconvolution of x is autoconv(1) + a * (autoconv(1)*autoconv(t) + autoconv(t)*autoconv(1)) + a^2 * autoconv(t).
    #    Actually, autoconv(a+b) = a^2 autoconv(1) + 2ab autoconv(1,t) + b^2 autoconv(t).
    #    If we use x = 1 + t (where t is +/-1), then:
    #    sum(x) = N.
    #    autoconv(x) = autoconv(ones) + 2*autoconv(ones, t) + autoconv(t).
    #    autoconv(ones) is a triangle peaking at N.
    #    autoconv(t) for Thue-Morse has low peaks.
    #    The cross term autoconv(ones, t) is the cumulative sum of t, which is also low for Thue-Morse.
    #    
    #    So, x = 1 + t_i where t_i is a Thue-Morse-like sequence scaled to have small amplitude 
    #    might be a good start. Or simply x_i = 1 + 0.5 * t_i.
    
    # Generate Thue-Morse sequence of length N
    # t_0 = 0, t_n = t_{n//2} XOR (n % 2)
    t = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        t[i] = t[i // 2] ^ (i % 2)
    
    # Map to {-1, 1}
    t_signed = 1 - 2 * t  # 0 -> 1, 1 -> -1
    
    # Initial candidate: x = 1 + 0.5 * t_signed
    # This ensures sum(x) is close to N (actually exactly N if N is even, or close).
    # Let's check sum.
    # If N=400, sum(t_signed) might be 0. So sum(x) = 400.
    
    x = 1.0 + 0.5 * t_signed
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Also try a few other initializations:
    # 1. All ones (baseline)
    x_ones = np.ones(N)
    sc_ones = _score(x_ones)
    if sc_ones < best_score:
        best_score = sc_ones
        best_x = x_ones.copy()
        
    # 2. Random smooth
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        # Low frequency sine
        freq = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2*np.pi)
        x_sin = 1.0 + 0.5 * np.sin(2 * np.pi * freq * np.arange(N) + phase)
        sc_sin = _score(x_sin)
        if sc_sin < best_score:
            best_score = sc_sin
            best_x = x_sin.copy()
            
    # 3. Local Search (Vectorized Coordinate Descent)
    # We will perform coordinate descent on the real-valued sequence.
    # To make it fast, we update one element at a time and recompute the score.
    # Recomputing the full autoconvolution via FFT for each single element update is O(N log N).
    # With N=400, this is fast. 10,000 updates * ~50us = 0.5s.
    # We can afford many iterations.
    
    x = best_x.copy()
    sc = best_score
    
    # Precompute FFT size
    n = N
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
        
    # We can optimize the score calculation by only updating the affected parts of the autoconvolution?
    # No, FFT is global. But we can reuse the FFT of the sequence?
    # If we change x[i] by delta, F_new = F_old + delta * exp(-2j pi k i / size2).
    # Then A_new = ifft(F_new * conj(F_new)).
    # This is still an ifft, which is O(N log N).
    # So we can do many iterations.
    
    # Let's do 5000 iterations of coordinate descent.
    # In each iteration, pick a random index, try a few deltas, pick the best.
    
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
            
        idx = rng.integers(0, N)
        orig_val = x[idx]
        
        # Try a few candidate values
        # We want to move x[idx] to reduce the score.
        # The score is max(|A|) / S^2.
        # Changing x[idx] changes S and A.
        
        best_delta = 0.0
        best_sc = sc
        
        # Heuristic: try values around current value
        # Step size can be adaptive
        step_size = 0.1 * (1 + 0.5 * np.sin(iteration * 0.01)) # Just to vary
        
        candidates = [orig_val + step_size, orig_val - step_size, 
                      orig_val + 0.5*step_size, orig_val - 0.5*step_size,
                      orig_val + 0.1*step_size, orig_val - 0.1*step_size]
        
        # Also try snapping to 0.5 or 1.5 if close? No, continuous is fine.
        
        for delta in candidates:
            if delta < -10 or delta > 10: # Bounds
                continue
            x[idx] = delta
            sc_new = _score(x)
            if sc_new < best_sc:
                best_sc = sc_new
                best_delta = delta
        
        x[idx] = best_delta
        sc = best_sc
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            
    return best_x
# EVOLVE-BLOCK-END
