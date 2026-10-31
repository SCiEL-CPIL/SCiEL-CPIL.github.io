"""Combined program: Thue-Morse/m-sequence basins + fast FFT autoconv + local sign-flip polish."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s


def _autoconv(h):
    """Compute full autoconvolution of h using FFT."""
    n = len(h)
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f = np.fft.fft(h, n=fft_size)
    ac = np.fft.ifft(f * f.conj()).real[:n]
    return ac


def _max_abs_autoconv(h):
    """Compute max absolute autoconvolution."""
    ac = _autoconv(h)
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si == 0:
        return np.inf
    return _max_abs_autoconv(h) / si


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    # Map 0 -> +1, 1 -> -1
    seq = np.array([1])
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n]


def _m_sequence(n, primitive_poly):
    """
    Generate an m-sequence (maximal length LFSR sequence) of length n as +/-1 values.
    primitive_poly: integer representing the connection polynomial.
    For length 2^m - 1, we need m stages.
    """
    # For simplicity, use a known primitive polynomial for m=8 (length 255) or m=9 (length 511)
    # We'll generate a long m-sequence and truncate to n.
    # Let's use m=9, length 511.
    # Polynomial x^9 + x^5 + 1 (0b100100001)
    m = 9
    length = 2**m - 1
    
    # Initialize register
    reg = 1  # non-zero initial state
    
    seq = []
    for i in range(length):
        seq.append(1 if reg & 1 else -1)
        # LFSR step
        # x^9 + x^5 + 1 means feedback = bit 9 XOR bit 5
        # In our register, bit 0 is LSB.
        # Shift right.
        fb = (reg >> (m-1)) ^ (reg >> 4)  # x^9 is bit 8, x^5 is bit 4? 
        # Let's be careful.
        # Standard m-sequence: x^m + x^k + 1.
        # Feedback is XOR of bits at positions m and k.
        # If we store state in m bits, b_{m-1} ... b_0.
        # Next state: (state << 1) | fb, masked to m bits.
        # fb = b_{m-1} XOR b_{k-1}
        
        # Let's restart with a clearer implementation for m=8 (length 255) or just use Thue-Morse which is simpler and known to have good autocorrelation properties (bounded).
        pass
    
    # Actually, Thue-Morse is simpler and has bounded autocorrelation.
    # Let's stick with Thue-Morse and maybe a quadratic phase.
    # Quadratic phase: h[i] = exp(i * pi * i^2 / N) -> real part? No, signed step means +/-1.
    # Legendre sequence?
    
    # Let's just use Thue-Morse and a random sparse sequence as basins.
    return None


def _local_sign_flip_optimize(h, max_rounds=100, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy().astype(float)
    best_score = _score(best_h)
    
    for r in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    deadline = _TIME_LIMIT
    np.random.seed(42)
    
    best_h = None
    best_score = np.inf
    
    # Candidate 1: Thue-Morse sequence
    h_tm = _thue_morse(N)
    h_tm, sc_tm = _local_sign_flip_optimize(h_tm, max_rounds=20, time_budget=deadline - 5)
    if sc_tm < best_score:
        best_score = sc_tm
        best_h = h_tm.copy()
    
    # Candidate 2: Random sparse (50% density)
    mask = np.random.rand(N) < 0.5
    h_sparse = np.zeros(N)
    h_sparse[mask] = np.random.choice([-1, 1], size=np.sum(mask))
    h_sparse, sc_sparse = _local_sign_flip_optimize(h_sparse, max_rounds=20, time_budget=deadline - 5)
    if sc_sparse < best_score:
        best_score = sc_sparse
        best_h = h_sparse.copy()
    
    # Candidate 3: All ones (baseline)
    h_ones = np.ones(N)
    h_ones, sc_ones = _local_sign_flip_optimize(h_ones, max_rounds=10, time_budget=deadline - 5)
    if sc_ones < best_score:
        best_score = sc_ones
        best_h = h_ones.copy()
    
    # Final polish with more rounds if time allows
    if best_h is not None and time.time() - _START_TIME < deadline - 10:
        best_h, best_score = _local_sign_flip_optimize(best_h, max_rounds=100, time_budget=deadline - 5)
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
