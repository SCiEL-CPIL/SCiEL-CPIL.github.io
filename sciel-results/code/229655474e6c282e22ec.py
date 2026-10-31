import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

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
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(a, p):
    """
    Compute the Legendre symbol (a/p).
    Returns 1 if a is a quadratic residue mod p, -1 if non-residue, 0 if a % p == 0.
    """
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0  # Should not happen for prime p

def _generate_legendre_sequence(p, N):
    """
    Generate a sequence of length N based on the Legendre symbol of a prime p.
    h[i] = Legendre(i+1, p) for i in range(N).
    If Legendre is 0, map to 1.
    """
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
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
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Use Legendre symbol sequences as initializations (algebraic structure).
    # 2. Use diverse other initializations (random, cosine, blocks).
    # 3. Apply local sign-flip optimization to each.
    # 4. Keep the best.
    
    n_restarts = 200  # Try many, break on deadline
    
    # Pre-compute Legendre sequences for a few primes near N
    # Primes near 400: 397, 401, 389, 409
    primes_to_try = [397, 401, 389, 409, 383, 419]
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        if r < len(primes_to_try):
            # Legendre symbol sequence
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r % 5 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Low frequency cosine: h[i] = sign(cos(2*pi*k*i/N))
            k = (r // 5) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
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
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
