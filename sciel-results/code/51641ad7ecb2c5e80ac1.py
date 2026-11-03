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
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _muller_rabin(n, a, d, r):
    """Check if n is prime using one base of Miller-Rabin."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
    return False

def _is_prime(n):
    """Deterministic primality test for n < 3,317,044,064,679,887,385,961,981."""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 3.3e24
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if not _muller_rabin(n, a, d, r):
            return False
    return True

def _next_prime(n):
    """Find the next prime >= n."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime(n):
        n += 2
    return n

def _construct_zadoff_chu_prime(n, prime, root=1):
    """
    Construct a Zadoff-Chu sequence of length n using a prime modulus.
    h[k] = exp(2j * pi * root * k^2 / prime)
    We return the real part for the signed step function.
    """
    k = np.arange(n)
    # Ensure k^2 is computed correctly modulo prime
    # For large primes, k^2 might overflow int64 if k is large, but here n=400, so k^2 < 160000, safe.
    phase = 2.0 * np.pi * root * (k * k) / prime
    # The sequence is typically complex. The task asks for "signed step heights".
    # The current program returns real values.
    # Let's return the real part, or perhaps the full complex magnitude isn't allowed?
    # "sequence of signed step heights" implies real numbers.
    # Zadoff-Chu sequences have constant modulus 1.
    # Using the real part: cos(2*pi*root*k^2/prime)
    return np.cos(phase)

def _construct_zadoff_chu_complex(n, prime, root=1):
    """
    Construct the complex Zadoff-Chu sequence.
    """
    k = np.arange(n)
    phase = 2.0 * np.pi * root * (k * k) / prime
    return np.exp(1j * phase)

def _gen_golden_ratio(n, seed=0):
    """
    Generate a sequence based on the golden ratio rotation, mapped to signs.
    This creates a low-discrepancy sequence of signs.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    x = np.arange(n)
    # Rotation on unit circle
    angles = 2.0 * np.pi * (x / phi + seed)
    # Map to signs: positive if cos > 0, negative otherwise
    # Or use the fractional part to determine sign
    # Let's use the real part of exp(i * angle)
    return np.cos(angles)

def _gen_random_smooth(n, rng, window_size=5):
    """
    Generate a random smooth sequence.
    """
    h = rng.standard_normal(n)
    kernel = np.ones(window_size) / window_size
    h = np.convolve(h, kernel, mode='same')
    return h

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
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

def _local_search_swap(h, max_passes=5, deadline=None):
    """
    Local optimization: try swapping signs of pairs of elements.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, N):
                if deadline is not None and time.time() > deadline:
                    break
                # Flip both i and j
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

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Zadoff-Chu sequences with various prime moduli
    # We want primes slightly larger than N to avoid periodicity issues within the sequence length
    # Or exactly N if N is prime (it's not).
    # Let's try primes around N and larger.
    primes_to_try = []
    p = _next_prime(N)
    for _ in range(10):
        primes_to_try.append(p)
        p = _next_prime(p + 2)
    
    for prime in primes_to_try:
        # Try different roots
        for root in [1, 2, 3]:
            if root < prime:
                inits.append(_construct_zadoff_chu_prime(N, prime, root))
    
    # 3. Golden ratio sequences
    for seed in [0.0, 0.1, 0.2, 0.3]:
        inits.append(_gen_golden_ratio(N, seed))
    
    # 4. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(10):
        inits.append(_gen_random_smooth(N, rng, window_size=3 + i % 5))
    
    # 5. Chirp sequences (quadratic phase)
    for scale in [1, 2, 3, 5, 10]:
        x = np.arange(N)
        phase = 2.0 * np.pi * scale * x**2 / (N**2)
        inits.append(np.cos(phase))
    
    # Optimization loop: For each initial guess, perform local search
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape?
        # The score is invariant to scaling: max|ac(c*h)| / (c*sum h)^2 = c^2 * max|ac(h)| / c^2 * (sum h)^2 = score(h)
        # So scaling doesn't change the score. We can work with the raw values.
        
        # Local search 1: Single flips
        if time.time() < deadline - 15:
            h_refined, val_refined = _local_search_flip(h_init, max_passes=3, deadline=deadline-10)
            if val_refined < best_score:
                best_score = val_refined
                best_h = h_refined.copy()
        
        # Local search 2: Pair flips (more expensive, do less often)
        if time.time() < deadline - 5:
            h_refined2, val_refined2 = _local_search_swap(h_refined, max_passes=2, deadline=deadline-2)
            if val_refined2 < best_score:
                best_score = val_refined2
                best_h = h_refined2.copy()
        
        # Also try refining the best_h found so far
        if time.time() < deadline - 5:
            h_best_refined, val_best_refined = _local_search_flip(best_h, max_passes=2, deadline=deadline-2)
            if val_best_refined < best_score:
                best_score = val_best_refined
                best_h = h_best_refined.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
