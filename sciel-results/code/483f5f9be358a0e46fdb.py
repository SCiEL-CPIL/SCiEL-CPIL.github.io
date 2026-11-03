"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _fast_local_search(h, deadline):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT for fast autoconvolution.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    # Precompute FFT size
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # We can optimize the search by checking if flipping improves the peak
    # But for simplicity and robustness, we do a standard flip search.
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
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

def _generate_initial_candidates():
    """Generate diverse initial binary sequences."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Random binary sequences
    for _ in range(10):
        h = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h)
        
    # 2. Structured sequences (e.g., chirps, m-sequences approximations)
    # Chirp-like
    x = np.arange(N)
    h_chirp = np.sign(np.cos(np.pi * x**2 / N))
    candidates.append(h_chirp)
    
    # 3. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(h_alt)
    
    # 4. Block patterns
    for block_size in [4, 8, 16]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        candidates.append(h_block)
        
    return candidates

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)  # Baseline: constant sequence (score will be high, but valid)

    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # Optimize each candidate
    for h_init in candidates:
        if time.time() > deadline - 5:
            break
            
        # Ensure sum is not zero (for binary ±1, sum is integer, so if N is even, sum can be 0)
        # If sum is 0, the objective is undefined (division by zero).
        # We can flip one bit to make sum non-zero.
        s = np.sum(h_init)
        if abs(s) < 1e-9:
            h_init[0] = -h_init[0]
            
        h_opt, val_opt = _fast_local_search(h_init, deadline - 2)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    # Final check
    return best_h

# EVOLVE-BLOCK-END
