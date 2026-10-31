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
        return 0

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

def _local_sign_flip_incremental(h, max_passes=20, deadline=None):
    """
    Local optimization with O(N) incremental autoconvolution updates.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    
    # Initial autoconvolution
    ac = np.convolve(h, h, mode='full')
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    
    max_abs_ac = np.max(np.abs(ac))
    best_val = max_abs_ac / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            old_h_i = h[i]
            delta = -2.0 * old_h_i
            
            # Apply update to ac
            # ac[k] += 2 * delta * h[k-i] for k in [i, i+n)
            # Note: h is still the old h
            ac[i : i+n] += 2.0 * delta * h
            # ac[2*i] += delta^2
            if 2*i < size:
                ac[2*i] += delta * delta
            
            s_new = s + delta
            if abs(s_new) < 1e-6:
                # Revert
                ac[i : i+n] -= 2.0 * delta * h
                if 2*i < size:
                    ac[2*i] -= delta * delta
                continue
            
            max_abs_new = np.max(np.abs(ac))
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                h[i] = -old_h_i
                s = s_new
                max_abs_ac = max_abs_new
                improved = True
            else:
                # Revert
                ac[i : i+n] -= 2.0 * delta * h
                if 2*i < size:
                    ac[2*i] -= delta * delta
        
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    n_restarts = 100
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r < len(primes_to_try):
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r % 5 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            k = (r // 5) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 2:
            x = np.arange(N)
            freq = 1.0 + (r // 5) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2*N)))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        else:
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Allow more passes due to speedup
        h_opt, val_opt = _local_sign_flip_incremental(h, max_passes=50, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
