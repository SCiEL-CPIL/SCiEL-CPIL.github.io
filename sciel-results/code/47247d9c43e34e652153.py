"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_max_abs(h):
    """Compute max |autoconvolution[k]| for k=1..N-1 using FFT."""
    n = len(h)
    # Next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is DC (sum of squares), we want max over lags 1..n-1
    if n > 1:
        return float(np.max(np.abs(acf[1:n])))
    return 0.0

def _legendre_sequence(n):
    """Generate Legendre sequence of length n (quadratic residues mod p)."""
    # Find smallest prime p >= n
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = i + 1
        if x == 1:
            seq[i] = 1.0
        else:
            # Legendre symbol (x/p)
            # Use Euler's criterion: x^((p-1)/2) mod p
            legendre = pow(x, (p - 1) // 2, p)
            if legendre == 1:
                seq[i] = 1.0
            elif legendre == p - 1:
                seq[i] = -1.0
            else:
                seq[i] = 0.0  # shouldn't happen for prime p
    return seq

def _m_sequence(n):
    """Generate an m-sequence of length 2^m - 1, then truncate/extend to n."""
    # Find m such that 2^m - 1 >= n
    m = 1
    while (1 << m) - 1 < n:
        m += 1
    
    # Use a known primitive polynomial for m
    # For simplicity, use a simple LFSR with primitive polynomial
    # x^m + x^(m-1) + ... + x + 1 is not always primitive
    # Use known primitive polynomials
    primitive_pols = {
        2: 0b11, 3: 0b101, 4: 0b1001, 5: 0b10011,
        6: 0b10101, 7: 0b10001, 8: 0b10011, 9: 0b10011,
        10: 0b101011, 11: 0b10001, 12: 0b10011,
        13: 0b10001, 14: 0b101011, 15: 0b10001,
        16: 0b10011, 17: 0b10001, 18: 0b101011,
        19: 0b10001, 20: 0b10011, 21: 0b10001,
        22: 0b101011, 23: 0b10001, 24: 0b10011,
        25: 0b10001, 26: 0b101011, 27: 0b10001,
        28: 0b10011, 29: 0b10001, 30: 0b101011,
    }
    
    if m not in primitive_pols:
        # Fallback to random
        return np.random.choice([-1.0, 1.0], size=n)
    
    poly = primitive_pols[m]
    # Initialize LFSR
    state = 1
    seq = []
    length = (1 << m) - 1
    
    for _ in range(length):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        # Shift and feedback
        feedback = (state & 1)
        for i in range(m - 1, 0, -1):
            bit = (state & (1 << i)) >> i
            next_bit = bit
            if (poly & (1 << i)) != 0:
                next_bit ^= feedback
            state = (state << 1) & ((1 << m) - 1)
            state |= next_bit
        state = (state & ((1 << (m-1)) - 1)) | (feedback << (m-1))
    
    # Truncate or extend to n
    if len(seq) >= n:
        return np.array(seq[:n], dtype=np.float64)
    else:
        # Repeat and truncate
        repeats = (n // len(seq)) + 1
        extended = np.tile(np.array(seq, dtype=np.float64), repeats)
        return extended[:n]

def _random_sequence(n, seed=None):
    """Generate random ±1 sequence."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)

def _local_search(h, max_iters=5000, time_limit=100.0):
    """Greedy local search to minimize max |autocorrelation|."""
    best_h = h.copy()
    best_score = _autoconvolution_max_abs(best_h)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random position
        idx = np.random.randint(len(h))
        h_new = best_h.copy()
        h_new[idx] *= -1.0
        
        score_new = _autoconvolution_max_abs(h_new)
        if score_new < best_score:
            best_h = h_new
            best_score = score_new
    
    return best_h, best_score

def _simulated_annealing(h, time_limit=120.0):
    """Simulated annealing to minimize max |autocorrelation|."""
    current_h = h.copy()
    current_score = _autoconvolution_max_abs(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 10.0  # initial temperature
    cooling = 0.999
    
    while time.time() - start_time < time_limit - 5.0:  # 5s safety margin
        T *= cooling
        if T < 0.01:
            break
        
        # Try flipping a random position
        idx = np.random.randint(len(h))
        h_new = current_h.copy()
        h_new[idx] *= -1.0
        
        score_new = _autoconvolution_max_abs(h_new)
        delta = score_new - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = h_new
            current_score = score_new
            if score_new < best_score:
                best_h = current_h.copy()
                best_score = score_new
    
    return best_h, best_score

def construct():
    """Construct a signed step function with low autoconvolution peak."""
    start_time = time.time()
    time_limit = 120.0
    
    # Generate candidate sequences
    candidates = []
    
    # Legendre sequence
    try:
        candidates.append(_legendre_sequence(N))
    except:
        pass
    
    # M-sequence
    try:
        candidates.append(_m_sequence(N))
    except:
        pass
    
    # Random sequences
    for seed in [42, 123, 456]:
        candidates.append(_random_sequence(N, seed=seed))
    
    # Evaluate initial candidates
    best_h = None
    best_score = float('inf')
    for h in candidates:
        score = _autoconvolution_max_abs(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search from best candidate
    best_h, best_score = _local_search(best_h, max_iters=2000, time_limit=30.0)
    
    # Simulated annealing if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20.0:
        sa_time = min(60.0, time_limit - elapsed - 10.0)
        if sa_time > 10.0:
            best_h, sa_score = _simulated_annealing(best_h, time_limit=sa_time)
            if sa_score < best_score:
                best_score = sa_score
    
    # Final local search polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10.0:
        polish_time = min(20.0, time_limit - elapsed - 5.0)
        if polish_time > 5.0:
            best_h, polish_score = _local_search(best_h, max_iters=5000, time_limit=polish_time)
            if polish_score < best_score:
                best_score = polish_score
    
    return best_h
# EVOLVE-BLOCK-END
