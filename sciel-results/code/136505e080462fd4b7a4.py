"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with N=400 intervals on [-1/4, 1/4]
    that minimizes the autoconvolution upper bound (C3).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def compute_acf(s):
        """Compute aperiodic autocorrelation using FFT."""
        n = len(s)
        size = 1
        while size < 2 * n:
            size *= 2
        s_pad = np.zeros(size)
        s_pad[:n] = s
        S = np.fft.fft(s_pad, size)
        acf_full = np.fft.ifft(S * np.conj(S)).real
        # ACF at lag k for k=0..n-1: acf_full[k]
        return acf_full[:n]
    
    def max_abs_acf(s):
        """Max |ACF| at non-zero lags."""
        acf = compute_acf(s)
        return np.max(np.abs(acf[1:]))
    
    def sum_sq_acf(s):
        """Sum of squared ACF at non-zero lags."""
        acf = compute_acf(s)
        return np.sum(acf[1:]**2)
    
    # Try multiple initialization strategies
    candidates = []
    
    # Strategy 1: Legendre sequence (for prime p, use first N bits)
    # Legendre symbol: (a/p) = 1 if a is quadratic residue mod p, -1 otherwise
    def legendre_sequence(n, p):
        """Generate a Legendre sequence of length n using prime p."""
        seq = np.zeros(n, dtype=np.float64)
        for i in range(n):
            a = i + 1
            if a % p == 0:
                seq[i] = -1  # Convention for 0
            else:
                # Compute Legendre symbol (a/p)
                ls = pow(a, (p-1)//2, p)
                if ls == p-1:
                    seq[i] = -1
                else:
                    seq[i] = 1
        return seq
    
    # Find a prime >= N
    def is_prime(n):
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
    
    p = N
    while not is_prime(p):
        p += 1
    
    candidates.append(legendre_sequence(N, p))
    
    # Strategy 2: m-sequence (maximum length LFSR sequence)
    # For N=400, we can use a 9-bit LFSR (length 511) and take first 400
    def m_sequence(n, poly, init):
        """Generate m-sequence from LFSR."""
        # poly: list of tap positions (1-indexed from LSB)
        # init: initial state
        seq = []
        state = init
        for _ in range(n):
            seq.append(1 if state & 1 else -1)
            # Shift and compute new bit
            feedback = 0
            for tap in poly:
                feedback ^= (state >> (tap - 1)) & 1
            state = (state >> 1) | (feedback << 8)
        return np.array(seq, dtype=np.float64)
    
    # 9-bit m-sequence with polynomial x^9 + x^4 + 1
    # Taps at positions 9 and 4 (0-indexed: 8 and 3)
    candidates.append(m_sequence(N, [9, 4], 1))
    
    # Strategy 3: Random
    np.random.seed(42)
    candidates.append(np.random.choice([-1, 1], size=N).astype(np.float64))
    
    # Evaluate candidates
    best_seq = candidates[0]
    best_val = max_abs_acf(best_seq)
    for c in candidates[1:]:
        val = max_abs_acf(c)
        if val < best_val:
            best_val = val
            best_seq = c.copy()
    
    # Local search: simulated annealing
    current_seq = best_seq.copy()
    current_val = best_val
    best_seq_sa = current_seq.copy()
    best_val_sa = current_val
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    min_T = 0.1
    
    while T > min_T and time.time() < deadline - 5:
        # Pick a random position to flip
        i = np.random.randint(N)
        old_val = current_seq[i]
        current_seq[i] = -current_seq[i]
        new_val = max_abs_acf(current_seq)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val_sa:
                best_val_sa = new_val
                best_seq_sa = current_seq.copy()
        else:
            current_seq[i] = old_val
        
        T *= cooling
    
    if best_val_sa < best_val:
        best_seq = best_seq_sa
        best_val = best_val_sa
    
    return best_seq
# EVOLVE-BLOCK-END
