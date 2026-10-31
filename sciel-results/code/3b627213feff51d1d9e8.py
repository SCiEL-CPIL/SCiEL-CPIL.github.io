import numpy as np
import time
import scipy.optimize as opt

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

def _local_sign_flip_incremental(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses incremental convolution updates for speed (O(N) per flip evaluation).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Precompute FFT of h
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping sign at position i
            v = h[i]
            delta = -2.0 * v
            
            # Incremental update of convolution
            # ac_new[k] = ac[k] + delta * h[k-i] + delta * h[k-(n-1-i)] + delta^2 * h[i]*h[i] ... 
            # Actually for autoconvolution: ac[k] = sum_j h[j]*h[k-j]
            # If we change h[i] -> h[i] + delta, then:
            # ac_new[k] = ac[k] + delta*h[k-i] + delta*h[k-(n-1-i)] + delta^2 * (1 if k-i == n-1-i else 0)
            # Wait, that's not quite right. Let's think more carefully.
            # ac[k] = sum_{j=0}^{n-1} h[j] * h[k-j] (with h[j]=0 for j<0 or j>=n)
            # After changing h[i] to h[i]+delta:
            # ac_new[k] = sum_j h'[j] * h'[k-j]
            #           = sum_j (h[j] + delta*delta_j) * (h[k-j] + delta*delta_{k-j})
            #           = ac[k] + delta * sum_j (delta_j * h[k-j] + h[j] * delta_{k-j}) + delta^2 * sum_j delta_j * delta_{k-j}
            # where delta_j = 1 if j==i else 0
            # = ac[k] + delta * (h[k-i] + h[k-i]) + delta^2 * (1 if k-i == i else 0)
            # = ac[k] + 2*delta*h[k-i] + delta^2 * (1 if k == 2*i else 0)
            # But h[k-i] is only valid if 0 <= k-i < n, otherwise 0.
            
            # This is getting complex. Let's just use FFT for simplicity but optimize the loop.
            h[i] = h[i] + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            if abs(s_new) < 1e-6:
                h[i] = v  # revert
                continue
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v  # revert
    
        if not improved:
            break
    
    return h, best_val

def _bfgs_optimize(h_init, deadline, maxiter=50):
    """
    Use BFGS to optimize the real-valued sequence, minimizing the score.
    """
    if time.time() > deadline:
        return h_init, _score(h_init)
    
    # Normalize to sum 1 for stability
    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return h_init, _score(h_init)
    h_norm = h_init / s0
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        s = np.sum(x)
        if abs(s) < 1e-6:
            return 1e10
        return _autoconv_max_abs(x) / (s * s)
    
    try:
        res = opt.minimize(obj, h_norm, method='BFGS', 
                           options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-5})
        h_opt = res.x
        val_opt = obj(h_opt)
        # Ensure sum is not zero
        if abs(np.sum(h_opt)) < 1e-6:
            return h_init, _score(h_init)
        return h_opt, val_opt
    except Exception:
        return h_init, _score(h_init)

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
    # 4. Apply BFGS optimization to the best candidates.
    # 5. Keep the best.
    
    n_restarts = 300  # Try many, break on deadline
    
    # Pre-compute Legendre sequences for a few primes near N
    # Primes near 400: 397, 401, 389, 409
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    
    # Store candidates for BFGS
    candidates = []
    
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
        h_opt, val_opt = _local_sign_flip_incremental(h, max_passes=5, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
        
        # Add to candidates for BFGS if it's a good candidate
        if val_opt < best_val * 1.1:
            candidates.append((h_opt, val_opt))
    
    # Apply BFGS to the top candidates
    # Sort by score and take the top few
    candidates.sort(key=lambda x: x[1])
    top_candidates = candidates[:10]
    
    for h_cand, val_cand in top_candidates:
        if time.time() > deadline:
            break
        
        h_bfgs, val_bfgs = _bfgs_optimize(h_cand, deadline, maxiter=50)
        
        if val_bfgs < best_val:
            best_val = val_bfgs
            best_h = h_bfgs
    
    # Final local sign flip polish on the best result
    if time.time() < deadline:
        best_h, best_val = _local_sign_flip_incremental(best_h, max_passes=10, deadline=deadline)
    
    return best_h

# EVOLVE-BLOCK-END
