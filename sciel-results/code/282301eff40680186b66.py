"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    Uses full recompute via FFT which is fast for N=400.
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
    
    # Strategy:
    # 1. Generate diverse initializations:
    #    a. Random +/-1
    #    b. Low-frequency cosine waves (smooth)
    #    c. Block structures
    #    d. "Chirp" patterns
    #    e. Sparse signed sequences
    # 2. Apply local sign-flip optimization to each.
    # 3. Keep the best.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts to try. We will adjust this based on time.
    # With FFT-based scoring, one score is ~ O(N log N).
    # Local flip pass is N * O(N log N) = O(N^2 log N).
    # For N=400, this is fast.
    
    n_restarts = 100  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        if r % 6 == 0:
            # Random +/-1
            h = rng.choice([-1.0, 1.0], size=N)
        elif r % 6 == 1:
            # Low frequency cosine: h[i] = sign(cos(2*pi*k*i/N))
            # Try different k
            k = (r // 6) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            # Ensure non-zero
            h[h == 0] = 1.0
        elif r % 6 == 2:
            # Chirp-like: frequency increases
            x = np.arange(N)
            freq = 1.0 + (r // 6) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2*N)))
            h[h == 0] = 1.0
        elif r % 6 == 3:
            # Block structure: alternating blocks of random length
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = rng.integers(2, 20)
                sign = rng.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        elif r % 6 == 4:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = rng.integers(0, N//2)
            neg_idx = rng.choice(N, size=int(n_neg), replace=False)
            h[neg_idx] = -1.0
        else:
            # Sparse signed sequence with random density
            density = rng.uniform(0.1, 0.5)
            mask = rng.random(N) < density
            h = np.zeros(N)
            h[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            s = np.sum(h)
            if abs(s) < 1e-12:
                flip_idx = rng.choice(N, size=max(1, N // 20), replace=False)
                h[flip_idx] *= -1
                s = np.sum(h)
                if abs(s) < 1e-12:
                    h[0] = 1.0
        
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

    # Final polish: try perturbations on the best solution found
    if time.time() < deadline - 5:
        x = best_h.copy()
        nonzero_idx = np.where(x != 0)[0]
        
        for trial in range(200):
            if time.time() > deadline - 2:
                break
            
            if len(nonzero_idx) > 0 and rng.random() < 0.7:
                # Flip or modify a non-zero entry
                idx = rng.choice(nonzero_idx)
                orig = x[idx]
                
                if abs(orig) == 1.0:
                    # Binary: flip sign
                    x[idx] = -orig
                else:
                    # Sparse: try different magnitude or sign
                    x[idx] = rng.choice([-1.0, 1.0])
                
                new_val = _score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_h = x.copy()
                    nonzero_idx = np.where(x != 0)[0]
                else:
                    x[idx] = orig
            else:
                # Add or modify a zero entry
                zero_idx = np.where(x == 0)[0]
                if len(zero_idx) > 0:
                    idx = rng.choice(zero_idx)
                    x[idx] = rng.choice([-1.0, 1.0])
                    new_val = _score(x)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_h = x.copy()
                        nonzero_idx = np.where(x != 0)[0]
                    else:
                        x[idx] = 0

    # Ensure valid result
    if abs(np.sum(best_h)) < 1e-6:
        best_h = np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
