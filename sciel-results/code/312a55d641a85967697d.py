"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_autocorr(h):
    """Compute the maximum absolute value of the linear autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    auto = np.fft.irfft(power, n=fft_size)
    if n <= 1:
        return 0.0
    peak = np.max(np.abs(auto[1:n]))
    return peak


def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _legendre_sequence(n, p):
    """Generate a Legendre sequence of length n using prime p."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = i + 1
        if a % p == 0:
            seq[i] = -1
        else:
            ls = pow(a, (p - 1) // 2, p)
            if ls == p - 1:
                seq[i] = -1
            else:
                seq[i] = 1
    return seq


def _m_sequence(n, poly, init, bits=9):
    """Generate m-sequence from LFSR."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if state & 1 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> (tap - 1)) & 1
        state = (state >> 1) | (feedback << (bits - 1))
    return np.array(seq, dtype=np.float64)


def _local_search(h, time_limit, allow_zero=True):
    """Local search: flip signs or toggle zero/non-zero to reduce peak autocorrelation."""
    n = len(h)
    h = h.copy()
    current_peak = _compute_peak_autocorr(h)
    best_peak = current_peak
    best_h = h.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try modifying each position
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = h[i]
            candidates = []
            if allow_zero:
                candidates = [-1.0, 0.0, 1.0]
            else:
                candidates = [-1.0, 1.0]
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                h[i] = new_val
                new_peak = _compute_peak_autocorr(h)
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()
                    break
                else:
                    h[i] = old_val
            
            if improved:
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_h, best_peak


def _sparse_init(rng, density):
    """Initialize with sparse non-zero values."""
    n = N
    h = np.zeros(n)
    num_nonzero = max(1, int(n * density))
    indices = rng.choice(n, size=num_nonzero, replace=False)
    h[indices] = rng.choice([-1.0, 1.0], size=num_nonzero)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_peak = np.inf
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Legendre sequences with different primes
    p = N
    while not _is_prime(p):
        p += 1
    candidates.append(_legendre_sequence(N, p))
    
    # Try a few other primes
    for offset in [2, 4, 6, 8, 10]:
        p2 = p + offset
        while not _is_prime(p2):
            p2 += 1
        candidates.append(_legendre_sequence(N, p2))
    
    # 2. m-sequences (9-bit LFSR, length 511, take first 400)
    # Different tap configurations
    mseq_polys = [
        [1, 8],
        [2, 8],
        [3, 8],
        [1, 2, 8],
        [1, 3, 8],
        [2, 3, 8],
    ]
    for poly in mseq_polys:
        for init in [1, 2, 3, 5]:
            candidates.append(_m_sequence(N, poly, init, bits=9))
    
    # 3. Sparse random initializations
    for density in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        for restart in range(3):
            candidates.append(_sparse_init(rng, density))
    
    # 4. Pure random ±1
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate each candidate and keep the best few
    scored_candidates = []
    for cand in candidates:
        peak = _compute_peak_autocorr(cand)
        scored_candidates.append((peak, cand))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Local search on the top candidates
    top_k = min(10, len(scored_candidates))
    time_per_candidate = max(1.0, (deadline - start_time - 5.0) / top_k)
    
    for i in range(top_k):
        if time.time() >= deadline:
            break
        
        peak_init, h_init = scored_candidates[i]
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            break
        time_limit = min(remaining, time_per_candidate)
        
        h_opt, peak_opt = _local_search(h_init, time_limit, allow_zero=True)
        
        if peak_opt < best_peak:
            best_peak = peak_opt
            best_h = h_opt.copy()
    
    # Final polish on the best solution
    if best_h is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 0:
            h_polished, peak_polished = _local_search(best_h, remaining, allow_zero=True)
            if peak_polished < best_peak:
                best_peak = peak_polished
                best_h = h_polished
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
