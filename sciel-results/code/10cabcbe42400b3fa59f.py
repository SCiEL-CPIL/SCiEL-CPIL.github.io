"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Pad to next power of 2 for faster FFT
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses incremental update for speed where possible, but full recompute is safer for correctness.
    Given the small N=400, full recompute via FFT is fast enough for several passes.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping sign
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Structural Change: Use a "Variational" initialization strategy.
    # Instead of purely random or simple patterns, we start with smooth, low-frequency
    # functions that naturally have lower autoconvolution sidelobes (like window functions
    # or low-frequency chirps) and then refine.
    # The hypothesis is that the optimal shape is not just +/-1 but might benefit from
    # continuous values or specific smooth structures that are then quantized/refined.
    # However, the task asks for "signed step heights", implying discrete steps, but the
    # seed allows any real sequence. The inspirations use +/-1 mostly.
    # Let's stick to +/-1 for the final answer but use smarter initializations.
    
    # Strategy:
    # 1. Generate diverse initializations:
    #    a. Random +/-1
    #    b. Low-frequency cosine waves (smooth)
    #    c. Block structures
    #    d. "Chirp" patterns
    # 2. Apply local sign-flip optimization to each.
    # 3. Keep the best.
    
    np.random.seed(42)
    
    # Number of restarts to try. We will adjust this based on time.
    # With FFT-based scoring, one score is ~ O(N log N).
    # Local flip pass is N * O(N log N) = O(N^2 log N).
    # For N=400, this is fast.
    
    n_restarts = 200  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        if r % 5 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Low frequency cosine: h[i] = sign(cos(2*pi*k*i/N))
            # Try different k
            k = (r // 5) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            # Ensure non-zero
            h[h == 0] = 1.0
        elif r % 5 == 2:
            # Chirp-like: frequency increases
            x = np.arange(N)
            freq = 1.0 + (r // 5) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2*N)))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            # Block structure: alternating blocks of random length
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        else:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        
        # Ensure the sum is not zero (which would make score inf)
        s = np.sum(h)
        if abs(s) < N * 0.1:
            # If sum is too small, flip the whole sequence if it helps, or add bias?
            # Since we are restricted to signed steps, we can just flip if sum < 0
            if s < 0:
                h = -h
            # If still small, we might need to force more +1s
            # But local optimization should handle this.
            # Let's just ensure it's not exactly zero by flipping one element if needed
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
