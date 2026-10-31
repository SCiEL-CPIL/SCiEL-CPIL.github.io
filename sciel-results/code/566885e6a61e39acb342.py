"""Optimized program for signed autocorrelation minimum peak-to-sum-ratio."""
import numpy as np
import time

N = 400

def compute_objective(x):
    """Compute max(|autoconv|) / (sum(x))^2"""
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    autoconv = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
    peak = np.max(np.abs(autoconv))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_x = None
    best_val = 1e10
    rng = np.random.default_rng(42)
    
    def try_candidate(x):
        nonlocal best_x, best_val
        val = compute_objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        return val
    
    # All ones baseline
    try_candidate(np.ones(N))
    
    # Try random binary sequences with bias toward +1 to keep |sum| large
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Bias toward +1
        p_plus = rng.uniform(0.6, 0.98)
        x = rng.choice([-1.0, 1.0], size=N, p=[1-p_plus, p_plus])
        
        val = try_candidate(x)
        
        # Short local search
        for it in range(300):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            old = x[idx]
            x[idx] = -old
            new_val = compute_objective(x)
            if new_val < val:
                val = new_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
            else:
                x[idx] = old
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
