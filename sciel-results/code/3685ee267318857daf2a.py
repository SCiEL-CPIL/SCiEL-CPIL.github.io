import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
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
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _generate_chirp_sequence(freq, N):
    """Generate a signed sequence from a quadratic phase chirp."""
    x = np.arange(N, dtype=np.float64)
    phase = np.pi * freq * x * x / N
    h = np.sign(np.cos(phase))
    h[h == 0] = 1.0
    return h

def _optimize_chirp_freq(freq_init, N, deadline):
    """Optimize chirp frequency using local search."""
    best_h = _generate_chirp_sequence(freq_init, N)
    best_val = _score(best_h)
    
    # Try perturbations around the initial frequency
    for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.001, -0.001]:
        if time.time() > deadline:
            break
        freq = freq_init + delta
        h = _generate_chirp_sequence(freq, N)
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h
    
    return best_h, best_val

def _local_search(h, deadline=None, max_passes=20):
    """
    Local optimization with single-flip and pair-flip moves.
    Returns (h_optimized, best_value).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Compute initial score
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single flip moves
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            
            # Try flipping position i
            h[i] = v + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v
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
                h[i] = v
        
        # Pair flip moves (try flipping pairs to escape local minima)
        if not improved:
            # Try a few random pair flips
            for _ in range(min(50, n // 2)):
                if deadline is not None and time.time() > deadline:
                    break
                
                i, j = np.random.choice(n, 2, replace=False)
                vi, vj = h[i], h[j]
                di, dj = -2.0 * vi, -2.0 * vj
                
                h[i] = vi + di
                h[j] = vj + dj
                
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
                max_abs_new = np.max(np.abs(ac_new))
                s_new = s + di + dj
                
                if abs(s_new) < 1e-6:
                    h[i] = vi
                    h[j] = vj
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
                    h[i] = vi
                    h[j] = vj
        
        if not improved:
            break
    
    return h, best_val

def _polish_search(h, deadline, max_passes=50):
    """Final high-precision polish with more iterations and tighter tolerance."""
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try all single flips
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            h[i] = v + delta
            
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v
                continue
                
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-14:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v
        
        # Try triple flips for more complex moves
        if not improved and n >= 3:
            for _ in range(min(30, n // 3)):
                if deadline is not None and time.time() > deadline:
                    break
                
                i, j, k = np.random.choice(n, 3, replace=False)
                vi, vj, vk = h[i], h[j], h[k]
                di, dj, dk = -2.0 * vi, -2.0 * vj, -2.0 * vk
                
                h[i] = vi + di
                h[j] = vj + dj
                h[k] = vk + dk
                
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
                max_abs_new = np.max(np.abs(ac_new))
                s_new = s + di + dj + dk
                
                if abs(s_new) < 1e-6:
                    h[i] = vi
                    h[j] = vj
                    h[k] = vk
                    continue
                    
                val_new = max_abs_new / (s_new * s_new)
                
                if val_new < best_val - 1e-14:
                    best_val = val_new
                    H = H_new
                    ac = ac_new
                    max_abs_ac = max_abs_new
                    s = s_new
                    improved = True
                else:
                    h[i] = vi
                    h[j] = vj
                    h[k] = vk
        
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Pre-compute Legendre sequences for primes near N
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431, 367, 433, 359, 439, 353, 443, 349, 449, 347, 457]
    
    # Chirp frequencies to try (non-integer, irrational-like)
    chirp_freqs = [0.5, 0.7, 1.3, 1.7, 2.3, 2.7, 3.3, 3.7, 4.3, 4.7, 
                   5.3, 5.7, 6.3, 6.7, 7.3, 7.7, 8.3, 8.7, 9.3, 9.7,
                   0.1, 0.3, 0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1,
                   0.4, 0.6, 1.4, 1.6, 2.4, 2.6, 3.4, 3.6, 4.4, 4.6]
    
    n_restarts = 1000  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initializations
        if r < len(primes_to_try):
            # Legendre symbol sequences
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r < len(primes_to_try) + len(chirp_freqs):
            # Chirp sequences with various frequencies
            idx = r - len(primes_to_try)
            freq = chirp_freqs[idx]
            h = _generate_chirp_sequence(freq, N)
        elif r % 7 == 0:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Low frequency cosine
            k = (r // 7) % 15 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 2:
            # Chirp-like with random frequency
            x = np.arange(N)
            freq = 0.5 + np.random.random() * 5.0
            h = np.sign(np.cos(np.pi * freq * x * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 3:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 15)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 7 == 4:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N // 3)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 7 == 5:
            # Walsh-Hadamard-like pattern
            x = np.arange(N)
            k = (r // 7) % 8 + 1
            bits = np.zeros(N, dtype=int)
            for b in range(8):
                bits |= ((x >> b) & 1) << b
            h = np.where((bits & (1 << k)) == 0, 1.0, -1.0)
        else:
            # Quadratic phase
            x = np.arange(N)
            freq = 1.0 + (r // 7) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x / N + np.pi * x * x / (N * N)))
            h[h == 0] = 1.0
        
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_search(h, deadline=deadline, max_passes=8)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Final polish pass on the best solution
    if time.time() < deadline - 5.0:
        best_h, best_val = _polish_search(best_h, deadline=deadline, max_passes=50)
    
    return best_h

# EVOLVE-BLOCK-END
