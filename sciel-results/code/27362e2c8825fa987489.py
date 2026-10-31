"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p using Euler's criterion."""
    if n % p == 0:
        return 0
    return (n ** ((p - 1) // 2)) % p


def _legendre_sequence(n, p):
    """
    Generate a Legendre sequence of length n using prime p.
    Returns ±1 sequence of length n.
    """
    seq = np.zeros(n)
    for i in range(n):
        ls = _legendre_symbol(i + 1, p)
        if ls == 0:
            seq[i] = 1.0
        elif ls == 1:
            seq[i] = 1.0
        else:
            seq[i] = -1.0
    return seq


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
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
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _incremental_update(h, i, sign_change=True):
    """
    Update the autoconvolution incrementally when flipping h[i].
    Returns the new autoconvolution array and the change in sum.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Current FFT
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    
    # Flip h[i]
    h[i] = -h[i]
    
    # Recompute FFT (still fast for N=400)
    H_new = np.fft.rfft(h, fsize)
    ac_new = np.fft.irfft(H_new * H_new, fsize)
    
    return ac_new


def _local_sign_flip_fast(h, max_passes=10, deadline=None):
    """
    Local optimization with incremental autoconvolution updates.
    Pre-compute FFT once per pass, update incrementally.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_val = _score(h)
    current_ac = np.fft.irfft(np.fft.rfft(h, fsize) ** 2, fsize)
    current_max = np.max(np.abs(current_ac))
    current_sum = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flip h[i]
            old_val = h[i]
            h[i] = -h[i]
            
            # Compute new autoconvolution
            H = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H * H, fsize)
            max_new = np.max(np.abs(ac_new))
            sum_new = current_sum + 2 * h[i]  # h[i] is now flipped
            
            if abs(sum_new) < 1e-12:
                new_val = 1e10
            else:
                new_val = max_new / (sum_new * sum_new)
            
            if new_val < best_val:
                best_val = new_val
                current_ac = ac_new
                current_max = max_new
                current_sum = sum_new
                improved = True
            else:
                h[i] = old_val  # revert
        
        if not improved:
            break
    
    return h, best_val


def _polish_pass(h, deadline=None):
    """
    High-precision polish: multiple passes of local sign flips with 
    occasional multi-bit flips to escape local minima.
    """
    h = h.copy()
    best_val = _score(h)
    
    for pass_num in range(15):
        if deadline is not None and time.time() > deadline:
            break
        
        # Single-bit flips
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        if not improved and pass_num > 3:
            # Try 2-bit flips
            if deadline is not None and time.time() > deadline:
                break
            for i in range(N - 1):
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = -h[i]
                h[i+1] = -h[i+1]
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = -h[i]
                    h[i+1] = -h[i+1]
            
            # Try 3-bit flips
            if not improved:
                for i in range(N - 2):
                    if deadline is not None and time.time() > deadline:
                        break
                    h[i] = -h[i]
                    h[i+1] = -h[i+1]
                    h[i+2] = -h[i+2]
                    new_val = _score(h)
                    if new_val < best_val:
                        best_val = new_val
                        improved = True
                    else:
                        h[i] = -h[i]
                        h[i+1] = -h[i+1]
                        h[i+2] = -h[i+2]
        
        if not improved:
            break
    
    return h, best_val


def _m_sequence(n, poly=0x13):
    """
    Generate an m-sequence (maximum length sequence) over GF(2).
    Returns a ±1 sequence of length n.
    """
    deg = poly.bit_length() - 1
    if deg < 2:
        deg = 4
        poly = 0x13
    
    state = 1
    period = (1 << deg) - 1
    bits = []
    
    for _ in range(period):
        bits.append(state & 1)
        feedback = 0
        for i in range(deg):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (deg - 1))
    
    mseq = np.array([1.0 if b == 1 else -1.0 for b in bits])
    
    if len(mseq) >= n:
        return mseq[:n]
    else:
        reps = (n + len(mseq) - 1) // len(mseq)
        return np.tile(mseq, reps)[:n]


def _chirp_sequence(n, k=1):
    """Generate a chirp-like sequence."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * k * x * (x + 1) / (2 * n)))
    h[h == 0] = 1.0
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Phase 0: Try m-sequences first
    mseq_polys = [0x13, 0x1B, 0x5D, 0x23, 0x25, 0x35, 0x45, 0x59, 0x65, 0x6D]
    for poly in mseq_polys:
        if time.time() > deadline:
            break
        h_m = _m_sequence(N, poly=poly)
        for sign in [1.0, -1.0]:
            h_try = sign * h_m
            val_try = _score(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
        h_ref, val_ref = _local_sign_flip_fast(h_m, max_passes=8, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 0b: Try Legendre sequences
    primes_to_try = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463]
    for p in primes_to_try:
        if time.time() > deadline:
            break
        h_leg = _legendre_sequence(N, p)
        for sign in [1.0, -1.0]:
            h_try = sign * h_leg
            val_try = _score(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
        h_ref, val_ref = _local_sign_flip_fast(h_leg, max_passes=8, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 1: Diverse initializations with local search
    n_restarts = 40
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 8 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 8 == 1:
            k = (r // 8) % 5 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 8 == 2:
            h = _chirp_sequence(N, k=1 + (r // 8) * 0.5)
        elif r % 8 == 3:
            h = np.random.uniform(-1, 1, size=N)
            h = np.sign(h)
            h[h == 0] = 1.0
        elif r % 8 == 4:
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(5, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 8 == 5:
            h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
            n_flips = np.random.randint(3, 15)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
        elif r % 8 == 6:
            p = primes_to_try[r % len(primes_to_try)]
            h = _legendre_sequence(N, p)
            n_flips = np.random.randint(3, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
        else:
            # Random with structure
            h = np.random.choice([-1.0, 1.0], size=N)
            # Add some block structure
            for _ in range(3):
                start = np.random.randint(0, N - 20)
                length = np.random.randint(5, 15)
                h[start:start+length] = np.random.choice([-1.0, 1.0])
        
        h_opt, val_opt = _local_sign_flip_fast(h, max_passes=6, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Phase 2: Refine best solution
    if time.time() < deadline - 15:
        h_refined, val_refined = _polish_pass(best_h, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
    
    # Phase 3: More diverse restarts if time permits
    if time.time() < deadline - 10:
        for r in range(20):
            if time.time() > deadline:
                break
            
            # Mix of algebraic and random
            if r % 3 == 0:
                h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
                n_flips = np.random.randint(1, 8)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                h[flip_idx] = -h[flip_idx]
            elif r % 3 == 1:
                p = primes_to_try[r % len(primes_to_try)]
                h = _legendre_sequence(N, p)
                n_flips = np.random.randint(1, 8)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                h[flip_idx] = -h[flip_idx]
            else:
                h = np.random.choice([-1.0, 1.0], size=N)
            
            h_opt, val_opt = _local_sign_flip_fast(h, max_passes=5, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
    
    # Final polish
    if time.time() < deadline - 1:
        h_final, val_final = _polish_pass(best_h, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h

# EVOLVE-BLOCK-END
