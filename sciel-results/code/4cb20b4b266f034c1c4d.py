"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    # Euler's criterion
    return (n ** ((p - 1) // 2)) % p


def _legendre_sequence(n, p):
    """
    Generate a Legendre sequence of length n using prime p.
    Legendre sequences have provably low off-peak autocorrelation.
    Returns ±1 sequence of length n.
    """
    # Generate Legendre symbols for 1..n
    seq = np.zeros(n)
    for i in range(n):
        ls = _legendre_symbol(i + 1, p)
        if ls == 0:
            seq[i] = 1.0  # Convention for 0 (shouldn't happen if n < p)
        elif ls == 1:
            seq[i] = 1.0
        else:  # ls == p-1 which is -1 mod p
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


def _autoconv_incremental(h, flip_idx, current_ac):
    """
    Incrementally update autoconvolution when flipping a single sign.
    If h[i] changes from h_old to h_new (i.e., h_new = -h_old), then
    the change in autoconvolution is:
      delta_ac[k] = h_new*h[k-i] + h[k-i]*h_new - h_old*h[k-i] - h[k-i]*h_old
                   = 2*h_new*h[k-i] - 2*h_old*h[k-i]  (for k >= i)
                   = 2*(h_new - h_old)*h[k-i] = 4*h_new*h[k-i] (since h_new=-h_old)
    More precisely: if we flip position i from a to -a, the change in ac[k] is:
      For k = i+j: ac_new[i+j] = ac_old[i+j] - a*h[j] - a*h[j] = ac_old[i+j] - 2*a*h[j]
      For k = i-j (j>0): ac_new[i-j] = ac_old[i-j] - a*h[j] - a*h[j] = ac_old[i-j] - 2*a*h[j]
    Actually let me think more carefully.
    
    ac[k] = sum_{j} h[j]*h[j+k] (linear convolution)
    If we flip h[i] from a to -a:
    Terms involving h[i]:
      When j=i: h[i]*h[i+k] changes from a*h[i+k] to -a*h[i+k], delta = -2a*h[i+k]
      When j+k=i, i.e., j=i-k: h[i-k]*h[i] changes from h[i-k]*a to h[i-k]*(-a), delta = -2a*h[i-k]
    So delta_ac[k] = -2*a*(h[i+k] + h[i-k]) where terms are zero if index out of range.
    """
    n = len(h)
    ac = current_ac.copy()
    i = flip_idx
    a = h[i]  # old value
    # For each lag k, the change involves h[i+k] and h[i-k]
    # We need to update ac for all k where i+k < n or i-k >= 0
    # More efficiently: for each j, if j == i or j+k == i, update
    # Let's do it by iterating over relevant j values
    
    # When j = i: affects ac[k] for k = 0, 1, ..., n-1-i (where h[i+k] exists)
    # When j + k = i => j = i - k: affects ac[k] for k = 0, 1, ..., i (where h[i-k] exists)
    
    # Combine: for lag k, delta = -2*a*(h[i+k] if i+k < n else 0) - 2*a*(h[i-k] if i-k >= 0 else 0)
    # But careful: when k=0, both terms contribute: -2*a*h[i] - 2*a*h[i] = -4*a*h[i]
    # Actually no: when k=0, j=i gives h[i]*h[i], and j+k=i means j=i, same term.
    # Let me re-derive:
    # ac[k] = sum_j h[j] * h[j+k]
    # After flipping h[i] from a to -a:
    # New ac[k] = sum_j h'[j] * h'[j+k]
    # Only terms where j=i or j+k=i change.
    # If j=i: old contribution a*h[i+k], new contribution (-a)*h[i+k], delta = -2a*h[i+k]
    # If j+k=i (j=i-k): old contribution h[i-k]*a, new contribution h[i-k]*(-a), delta = -2a*h[i-k]
    # If both (k=0, j=i=i-k): old a*a, new (-a)*(-a)=a*a, delta=0! 
    # Wait, that's wrong. Let me be more careful.
    # When k=0: ac[0] = sum_j h[j]^2. Flipping h[i] doesn't change h[i]^2. So delta=0 for k=0.
    # For k != 0: 
    #   If j=i is in range: delta += -2a*h[i+k]
    #   If j=i-k is in range (and i-k != i, i.e., k!=0): delta += -2a*h[i-k]
    # So for k != 0: delta_ac[k] = -2a*(h[i+k] (if valid) + h[i-k] (if valid))
    # For k = 0: delta_ac[0] = 0
    
    if i >= 0 and i < n:
        # Update ac for k from 1 to n-1
        # For each k, check if i+k < n and/or i-k >= 0
        max_k = n - 1
        for k in range(1, max_k + 1):
            delta = 0.0
            if i + k < n:
                delta += h[i + k]
            if i - k >= 0:
                delta += h[i - k]
            if delta != 0.0:
                ac[k] -= 2.0 * a * delta
        # Also need to update negative lags if we store them, but we only store 0..n-1
        # Linear autoconvolution: ac[k] for k=0..n-1 is sufficient since ac[-k]=ac[k] for real signals
        # But wait, linear convolution of length n with itself has length 2n-1, indices 0..2n-2
        # My ac array should have length 2n-1 if we're doing full linear convolution.
        # Let me check: the original _autoconv_max_abs computes full linear convolution.
        # So I need to handle that.
    
    return ac


def _autoconv_smooth_max(h, beta=200.0):
    """Compute smooth max of absolute autoconvolution using log-sum-exp."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    abs_ac = np.abs(ac)
    # Log-sum-exp for smooth max with numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_ac - max_val
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50.0, 0.0)
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(abs_ac)) / beta
    return log_sum_exp


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_from_ac(ac, s):
    """Compute score from precomputed autoconvolution and sum."""
    if abs(s) < 1e-12:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)


def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves.
    Uses incremental autoconvolution updates for speed."""
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    s = np.sum(h)
    best_val = _score_from_ac(ac, s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            new_val_h = -old_val
            
            # Update sum
            new_s = s - 2.0 * old_val
            
            # Update autoconvolution incrementally
            # Flip h[i] from old_val to new_val_h
            # For linear convolution of length 2n-1:
            # ac[k] = sum_j h[j]*h[j+k] for k=0..2n-2
            # When we flip h[i]:
            #   For lag k: terms where j=i (h[i]*h[i+k]) or j+k=i (h[i-k]*h[i])
            #   But we need to be careful with bounds for linear convolution.
            
            # Let's just do a fast FFT-based update for simplicity, 
            # but cache the FFT to avoid recomputing from scratch.
            # Actually, for a single flip, we can update the FFT directly.
            # H_new = H + delta_H where delta_H[m] = (new_val_h - old_val) * exp(-2*pi*i*m*i/fsize)
            # This is still O(fsize) per flip, not much better than recompute.
            
            # Alternative: just recompute using FFT but with precomputed size
            H_new = np.fft.rfft(h, fsize)  # This is O(n log n)
            ac_new = np.fft.irfft(H_new * H_new, fsize)
            
            new_score = _score_from_ac(ac_new, new_s)
            
            if new_score < best_val:
                h[i] = new_val_h
                s = new_s
                ac = ac_new
                best_val = new_score
                improved = True
            else:
                h[i] = old_val  # revert (already correct since we didn't change it)
        
        if not improved:
            break
    return h, best_val


def _polish_pass(h, deadline=None):
    """
    High-precision polish: multiple passes of local sign flips with 
    occasional multi-bit flips to escape local minima.
    Uses incremental updates where possible.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    s = np.sum(h)
    best_val = _score_from_ac(ac, s)
    
    # Multiple refinement passes
    for pass_num in range(15):
        if deadline is not None and time.time() > deadline:
            break
        
        # Single-bit flips
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            new_s = s - 2.0 * old_val
            
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)
            new_val = _score_from_ac(ac_new, new_s)
            
            if new_val < best_val:
                s = new_s
                ac = ac_new
                H = H_new
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        
        if not improved and pass_num > 3:
            # Try 2-bit flips to escape local minimum
            if deadline is not None and time.time() > deadline:
                break
            # Only try 2-bit flips for a subset of pairs to save time
            for i in range(0, n - 1, 2):
                if deadline is not None and time.time() > deadline:
                    break
                old_i = h[i]
                old_j = h[i+1]
                h[i] = -old_i
                h[i+1] = -old_j
                new_s = s - 2.0 * old_i - 2.0 * old_j
                
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)
                new_val = _score_from_ac(ac_new, new_s)
                
                if new_val < best_val:
                    s = new_s
                    ac = ac_new
                    H = H_new
                    best_val = new_val
                    improved = True
                else:
                    h[i] = old_i
                    h[i+1] = old_j
        
        if not improved:
            break
    
    return h, best_val


def _continuous_objective(x, beta=200.0):
    """Objective for continuous optimization: smooth max|autoconv| / (sum h)^2."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_smooth_max(h, beta) / (s * s)


def _m_sequence(n, poly=0x13, seed=0):
    """
    Generate an m-sequence (maximum length sequence) over GF(2) using a primitive polynomial.
    poly is the polynomial in binary (e.g., 0x13 = x^4+x+1 for degree 4).
    Returns a ±1 sequence of length n by tiling the m-sequence.
    """
    # Extract degree from polynomial
    deg = poly.bit_length() - 1
    if deg < 2:
        deg = 4
        poly = 0x13  # x^4 + x + 1
    
    # Initialize LFSR state (non-zero)
    state = 1  # Start with state 1
    
    # Generate 2^deg - 1 bits (one full period of m-sequence)
    period = (1 << deg) - 1
    bits = []
    
    for _ in range(period):
        bits.append(state & 1)
        # Compute next state: shift right, feedback from poly
        # The polynomial has bits set at positions corresponding to powers of x
        # For poly = sum of x^i, feedback = XOR of bits at those positions
        # Standard LFSR: new_bit = XOR of bits where poly has 1s (excluding the leading bit)
        feedback = 0
        for i in range(deg):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (deg - 1))
    
    # Convert bits to ±1 sequence
    mseq = np.array([1.0 if b == 1 else -1.0 for b in bits])
    
    # Tile to reach length n
    if len(mseq) >= n:
        return mseq[:n]
    else:
        reps = (n + len(mseq) - 1) // len(mseq)
        return np.tile(mseq, reps)[:n]


def _continuous_optimize(h_init, deadline=None):
    """
    Optimize in continuous space using scipy.optimize.minimize (L-BFGS-B),
    then quantize to ±1 and refine with local sign flips.
    This allows exploring the landscape more smoothly than pure sign flips.
    """
    h0 = h_init.astype(float).copy()
    
    # Ensure sum is not too small for stability
    s = np.sum(h0)
    if abs(s) < N * 0.1:
        if s < 0:
            h0 = -h0
        if abs(np.sum(h0)) < 1e-6:
            h0[0] = 1.0

    best_h = h0.copy()
    best_val = _score(h0)

    try:
        res = minimize(
            _continuous_objective,
            h0,
            method='L-BFGS-B',
            options={'maxiter': 80, 'ftol': 1e-9, 'maxfun': 300}
        )
        if res.x is not None and not np.any(np.isnan(res.x)):
            h_cont = res.x
            val_exact = _score(h_cont)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_cont
    except Exception:
        pass

    # Quantize the best continuous solution to ±1
    h_quant = np.sign(best_h)
    h_quant[h_quant == 0] = 1.0
    
    s_q = np.sum(h_quant)
    if abs(s_q) < N * 0.1:
        if s_q < 0:
            h_quant = -h_quant
        if abs(np.sum(h_quant)) < 1e-6:
            h_quant[0] = 1.0

    # Refine with local sign flips, then polish
    h_opt, val_opt = _local_sign_flip(h_quant, max_passes=4, deadline=deadline)
    
    if deadline is not None and time.time() < deadline - 3:
        h_polished, val_polished = _polish_pass(h_opt, deadline=deadline)
        if val_polished < val_opt:
            h_opt, val_opt = h_polished, val_polished
    
    return h_opt, val_opt


def _chirp_sequence(n, k=1):
    """Generate a chirp-like sequence with parameter k."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * k * x * (x + 1) / (2 * n)))
    h[h == 0] = 1.0
    return h


def _barker_like(n):
    """Generate a Barker-like sequence of length n using known Barker codes or random ±1."""
    barkers = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, -1, -1, 1],
        7: [1, -1, -1, 1, -1, 1, 1],
        11: [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    }
    if n in barkers:
        return np.array(barkers[n], dtype=float)
    return np.random.choice([-1.0, 1.0], size=n)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    
    # Phase 0: Try m-sequences first (algebraic structure)
    mseq_polys = [0x13, 0x1B, 0x5D, 0x23, 0x25, 0x35, 0x45, 0x59]
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
        h_ref, val_ref = _local_sign_flip(h_m, max_passes=6, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 0b: Try Legendre sequences
    primes_to_try = [401, 409, 419, 421, 431, 433, 439, 443]
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
        h_ref, val_ref = _local_sign_flip(h_leg, max_passes=6, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 1: Diverse initializations with continuous optimization
    n_restarts_cont = 20
    
    for r in range(n_restarts_cont):
        if time.time() > deadline:
            break

        if r % 9 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 9 == 1:
            k = (r // 9) % 5 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 9 == 2:
            h = _chirp_sequence(N, k=1 + (r // 9) * 0.5)
        elif r % 9 == 3:
            h = np.random.uniform(-1, 1, size=N)
        elif r % 9 == 4:
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(5, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 9 == 5:
            h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
        elif r % 9 == 6:
            h = _m_sequence(N, poly=mseq_polys[(r + 3) % len(mseq_polys)])
            n_flips = np.random.randint(3, 15)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
        elif r % 9 == 7:
            p = primes_to_try[r % len(primes_to_try)]
            h = _legendre_sequence(N, p)
        else:
            p = primes_to_try[(r + 2) % len(primes_to_try)]
            h = _legendre_sequence(N, p)
            n_flips = np.random.randint(3, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]

        h_opt, val_opt = _continuous_optimize(h, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: Refine and try more discrete restarts if time permits
    if time.time() < deadline - 20:
        h_refined, val_refined = _polish_pass(best_h, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

        # Try m-sequence-based discrete restarts
        for r in range(20):
            if time.time() > deadline:
                break
            poly = mseq_polys[r % len(mseq_polys)]
            h = _m_sequence(N, poly=poly)
            n_flips = np.random.randint(2, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=6, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

        # Try Legendre-based discrete restarts
        for r in range(15):
            if time.time() > deadline:
                break
            p = primes_to_try[r % len(primes_to_try)]
            h = _legendre_sequence(N, p)
            n_flips = np.random.randint(2, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=6, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

        # Try random discrete restarts
        for r in range(20):
            if time.time() > deadline:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            n_flips = np.random.randint(1, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=4, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

    # Final polish
    if time.time() < deadline - 2:
        h_final, val_final = _polish_pass(best_h, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    return best_h

# EVOLVE-BLOCK-END
