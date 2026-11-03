"""Direct analytic construction for signed autocorrelation minimum task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _compute_score(h):
    """Compute the metric: max |autoconvolution| / (sum(h))^2."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    c = fftconvolve(h, h, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (S * S)


def _m_sequence(n):
    """Generate an m-sequence of length n (must be power of 2) mapped to {-1, +1}."""
    # Find next power of 2 >= n
    m = 1
    while m < n:
        m *= 2
    # Use a simple LFSR to generate m-sequence
    # For m-sequence of period 2^k - 1, we need k bits
    k = int(np.log2(m + 1))
    # Use a known primitive polynomial for small k
    # For k=8 (period 255), use polynomial x^8 + x^4 + x^3 + x^2 + 1
    if k == 8:
        taps = [7, 4, 3, 2, 0]  # positions for LFSR
    elif k == 7:
        taps = [6, 5, 4, 3, 0]
    elif k == 6:
        taps = [5, 3, 2, 1, 0]
    elif k == 5:
        taps = [4, 3, 2, 1, 0]
    else:
        # Fallback: use a simple pattern
        return np.sign(np.sin(np.arange(m) * 2 * np.pi / m * 3))[:n]
    
    # Generate LFSR sequence
    state = np.zeros(k, dtype=np.int32)
    state[0] = 1  # Initialize with non-zero state
    seq = []
    for _ in range(m):
        # Compute feedback
        fb = 0
        for t in taps:
            fb ^= state[t]
        # Shift
        for i in range(k-1, 0, -1):
            state[i] = state[i-1]
        state[0] = fb
        seq.append(1 if state[0] else -1)
    
    seq = np.array(seq[:n], dtype=np.float64)
    return seq


def _block_pattern(n, block_size):
    """Generate a block pattern: alternating +1 and -1 in blocks."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        h[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
    return h


def construct():
    """
    Uses simulated annealing with diverse starting points to find signed step
    heights minimizing max|autoconvolution| / (sum(h))^2.
    
    Structural change: instead of relying on the analytic constant-sequence
    solution, we perform numerical optimization over the space of signed
    sequences using simulated annealing, starting from multiple diverse
    initializations (m-sequences, block patterns, random).
    """
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 120s limit
    
    # Diverse starting points
    h = np.full(N, 1.0, dtype=np.float64)
    best_score = _compute_score(h)
    
    # Try m-sequence
    h_m = _m_sequence(N)
    s_m = _compute_score(h_m)
    if s_m < best_score:
        best_score = s_m
        h = h_m.copy()
    
    # Try block patterns
    for bs in [4, 8, 16, 32, 64]:
        h_b = _block_pattern(N, bs)
        s_b = _compute_score(h_b)
        if s_b < best_score:
            best_score = s_b
            h = h_b.copy()
    
    # Try random
    np.random.seed(42)
    h_r = np.random.choice([-1.0, 1.0], size=N)
    s_r = _compute_score(h_r)
    if s_r < best_score:
        best_score = s_r
        h = h_r.copy()
    
    # Simulated annealing
    current = h.copy()
    current_score = best_score
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.995
    np.random.seed(123)
    
    while time.time() < deadline:
        # Propose a flip
        idx = np.random.randint(N)
        current[idx] *= -1.0
        new_score = _compute_score(current)
        
        if new_score < current_score or np.random.random() < np.exp((current_score - new_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                h = current.copy()
        else:
            current[idx] *= -1.0  # Revert
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return h
# EVOLVE-BLOCK-END
