"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _fft_size(n):
    """Next power of 2 >= n."""
    size = 1
    while size < n:
        size *= 2
    return size


def _autoconv(h):
    """Compute linear autoconvolution of h using FFT."""
    n = len(h)
    size = _fft_size(2 * n - 1)
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)[:2 * n - 1]


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    return np.max(np.abs(_autoconv(h)))


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_fast(h, ac_cache=None):
    """Faster score using precomputed autoconv if available."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    if ac_cache is not None:
        return np.max(np.abs(ac_cache)) / (s * s)
    return _autoconv_max_abs(h) / (s * s)


def _local_sign_flip_fast(h, max_passes=5, deadline=None):
    """
    Local optimization with incremental autoconv updates.
    When we flip h[i] -> -h[i], the autoconv changes in a structured way.
    For autoconv A[k] = sum_j h[j]*h[j+k], flipping h[i] changes:
    A[k] for k where i or i+k is in range.
    
    Actually, let's use a simpler but still fast approach:
    Precompute the autoconv, and for each flip, update only the affected terms.
    
    For linear autoconv of length n: A[m] = sum_{j} h[j] * h[j+m] for m=0..n-1
    and A[n-1-m] = A[m] for the symmetric part (actually it's not symmetric in linear conv).
    
    Let me think more carefully. Linear autoconvolution:
    A[m] = sum_{j=0}^{n-1-m} h[j] * h[j+m]  for m = 0, 1, ..., n-1
    
    If we flip h[i] (multiply by -1), then:
    For each m, A[m] contains terms h[j]*h[j+m] where j or j+m equals i.
    - j = i: term is h[i]*h[i+m] (if i+m < n), so j ranges such that j=i, meaning m can be 0..n-1-i
    - j+m = i: term is h[j]*h[i] (if j >= 0 and j < n-m), so j = i-m, meaning m can be 0..min(i, n-1)
    
    Actually this gets complex. Let me just recompute the autoconv via FFT each time
    but use a smarter approach: batch the evaluations.
    
    Alternative: use the fact that for ±1 sequences, we can compute the score
    incrementally. But FFT is O(n log n) which for n=400 is very fast.
    
    Let me just do a straightforward approach but be smart about when to check deadline.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try each position
        for i in range(n):
            if deadline is not None and time.time() > deadline:
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


def _local_sign_flip_pair(h, max_passes=5, deadline=None):
    """
    Local optimization trying pairs of sign flips.
    This can escape local minima that single flips can't.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # First try single flips
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        # If no single flip improved, try a few random pair flips
        if not improved:
            for _ in range(50):
                if deadline is not None and time.time() > deadline:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                h[i] = -h[i]
                h[j] = -h[j]
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = -h[i]
                    h[j] = -h[j]
        
        if not improved:
            break
    return h, best_val


def _legendre_sequence(n, prime=None):
    """Generate a Legendre symbol sequence of length n."""
    if prime is None:
        def is_prime(m):
            if m < 2:
                return False
            if m == 2:
                return True
            if m % 2 == 0:
                return False
            for i in range(3, int(m**0.5) + 1, 2):
                if m % i == 0:
                    return False
            return True
        p = n
        while not is_prime(p):
            p += 1
        prime = p
    else:
        p = prime
    
    if p < 3:
        p = 3
    
    h = np.zeros(p)
    for k in range(1, p):
        ls = pow(k, (p - 1) // 2, p)
        if ls == 1:
            h[k] = 1.0
        elif ls == p - 1:
            h[k] = -1.0
        else:
            h[k] = 0.0
    
    h[0] = 1.0
    
    if len(h) >= n:
        return h[:n]
    else:
        reps = (n + len(h) - 1) // len(h)
        return np.tile(h, reps)[:n]


def _m_sequence(n, poly=0x13):
    """Generate an m-sequence over GF(2) and convert to ±1."""
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


def _continuous_objective(x, beta=200.0):
    """Smooth surrogate objective for continuous optimization."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_ac - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(abs_ac)) / beta
    return log_sum_exp / (s * s)


def _continuous_optimize(h_init, deadline=None):
    """Optimize in continuous space then quantize and refine."""
    h0 = h_init.astype(float).copy()
    
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
            options={'maxiter': 60, 'ftol': 1e-8, 'maxfun': 200}
        )
        if res.x is not None and not np.any(np.isnan(res.x)):
            val_exact = _score(res.x)
            if val_exact < best_val:
                best_val = val_exact
                best_h = res.x
    except Exception:
        pass

    h_quant = np.sign(best_h)
    h_quant[h_quant == 0] = 1.0
    
    s_q = np.sum(h_quant)
    if abs(s_q) < N * 0.1:
        if s_q < 0:
            h_quant = -h_quant
        if abs(np.sum(h_quant)) < 1e-6:
            h_quant[0] = 1.0

    h_opt, val_opt = _local_sign_flip_pair(h_quant, max_passes=5, deadline=deadline)
    
    return h_opt, val_opt


def _is_prime(m):
    """Check if m is prime."""
    if m < 2:
        return False
    if m == 2:
        return True
    if m % 2 == 0:
        return False
    for i in range(3, int(m**0.5) + 1, 2):
        if m % i == 0:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    
    # Collect Legendre primes near N
    legendre_primes = []
    for p in range(N - 15, N + 25):
        if p > 2 and _is_prime(p):
            legendre_primes.append(p)
    
    mseq_polys = [0x13, 0x1B, 0x5D, 0x23, 0x25, 0x35, 0x45, 0x59, 0x11D, 0x173]

    # Phase 0: Algebraic sequences with local refinement
    for p in legendre_primes:
        if time.time() > deadline:
            break
        h_leg = _legendre_sequence(N, prime=p)
        for sign in [1.0, -1.0]:
            h_try = sign * h_leg
            val_try = _score(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
        h_ref, val_ref = _local_sign_flip_pair(h_leg, max_passes=10, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

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
        h_ref, val_ref = _local_sign_flip_pair(h_m, max_passes=8, deadline=deadline)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref

    # Phase 1: Diverse continuous optimization restarts
    n_restarts_cont = 15
    for r in range(n_restarts_cont):
        if time.time() > deadline:
            break

        if r % 5 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            k = (r // 5) % 4 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 2:
            h = _chirp_sequence(N, k=1 + (r // 5) * 0.7)
        elif r % 5 == 3:
            p = legendre_primes[r % len(legendre_primes)] if legendre_primes else None
            h = _legendre_sequence(N, prime=p)
            n_flips = np.random.randint(3, 12)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
        else:
            h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
            n_flips = np.random.randint(2, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]

        h_opt, val_opt = _continuous_optimize(h, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: More discrete restarts with pair-flip local search
    if time.time() < deadline - 15:
        # Refine best with more passes
        h_refined, val_refined = _local_sign_flip_pair(best_h, max_passes=25, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined

        # Discrete restarts from algebraic structures
        for r in range(25):
            if time.time() > deadline:
                break
            if r % 3 == 0:
                p = legendre_primes[r % len(legendre_primes)] if legendre_primes else None
                h = _legendre_sequence(N, prime=p)
            elif r % 3 == 1:
                h = _m_sequence(N, poly=mseq_polys[r % len(mseq_polys)])
            else:
                h = np.random.choice([-1.0, 1.0], size=N)
            
            n_flips = np.random.randint(2, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip_pair(h, max_passes=6, deadline=deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

    # Phase 3: Final polish with many local passes
    if time.time() < deadline - 3:
        h_polished, val_polished = _local_sign_flip_pair(best_h, max_passes=40, deadline=deadline)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished

    return best_h

# EVOLVE-BLOCK-END
