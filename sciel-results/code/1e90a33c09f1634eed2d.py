"""
Optimized PSL minimization for signed step function with N=400 intervals.
Uses LFSR-based m-sequence construction combined with fast gradient optimization.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _generate_lfsr_sequence(length, seed=0):
    """
    Generate a binary sequence using a linear feedback shift register (LFSR)
    over GF(2) with a primitive polynomial. Map bits to {-1, 1}.
    
    For length 400, we use a primitive polynomial of degree m where 2^m - 1 >= 400.
    We take the first `length` bits of the m-sequence.
    """
    # Find smallest m such that 2^m - 1 >= length
    m = 1
    while (1 << m) - 1 < length:
        m += 1
    
    # Use known primitive polynomials for small m, or a simple one
    # For our purposes, we'll use a simple LFSR with a fixed tap pattern
    # that gives good pseudo-random properties
    
    rng = np.random.default_rng(seed)
    
    # Initialize register with random non-zero state
    state = np.zeros(m, dtype=np.int32)
    state[0] = 1  # Ensure non-zero
    
    # Generate sequence
    seq = []
    for _ in range(length):
        seq.append(state[0])
        # Compute feedback bit (simple tap: XOR of bits at positions 0 and m-1)
        feedback = state[0] ^ state[m - 1]
        # Shift right
        state = np.roll(state, -1)
        state[0] = feedback
    
    # Map 0 -> -1, 1 -> 1
    h = np.array([1.0 if b == 1 else -1.0 for b in seq])
    return h


def _generate_m_sequence(m, length, seed=0):
    """
    Generate a maximal-length sequence (m-sequence) of the given length.
    Uses a primitive polynomial for the LFSR.
    
    For m=9: 2^9-1 = 511 >= 400
    Primitive polynomial for m=9: x^9 + x^4 + 1 (taps at positions 0, 4, 9)
    """
    # Choose m such that 2^m - 1 >= length
    while (1 << m) - 1 < length:
        m += 1
    
    # Primitive polynomial taps for various m
    # Format: list of tap positions (0-indexed, excluding the output tap)
    primitive_polys = {
        9: [0, 4],      # x^9 + x^4 + 1
        10: [0, 3],     # x^10 + x^3 + 1
        11: [0, 2],     # x^11 + x^2 + 1
        12: [0, 4],     # x^12 + x^4 + 1
        13: [0, 4],     # x^13 + x^4 + 1
        14: [0, 5],     # x^14 + x^5 + 1
        15: [0, 1],     # x^15 + x + 1
    }
    
    if m not in primitive_polys:
        # Fallback: use simple taps
        primitive_polys[m] = [0, m // 2]
    
    taps = primitive_polys[m]
    
    # Initialize with non-zero state
    state = np.zeros(m, dtype=np.int32)
    state[0] = 1
    
    # Generate sequence
    seq = []
    for _ in range(length):
        seq.append(state[0])
        # Compute feedback
        feedback = state[0]
        for tap in taps:
            feedback ^= state[tap]
        # Shift right and insert feedback
        state = np.roll(state, -1)
        state[0] = feedback
    
    # Map 0 -> -1, 1 -> 1
    h = np.array([1.0 if b == 1 else -1.0 for b in seq])
    return h


def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    Returns (best_i, new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    # Build S matrix: S[i,k] = h[i-k] + h[i+k] with boundary handling
    S = np.zeros((n, n - 1), dtype=np.float64)
    for k in range(1, n):
        idx = k - 1
        # h[i-k] for i = k, k+1, ..., n-1
        S[k:, idx] += h[:n - k]
        # h[i+k] for i = 0, 1, ..., n-1-k
        S[:n - k, idx] += h[k:]
    
    # delta[i, k] = -2 * h[i] * S[i, k]
    delta = -2.0 * h[:, np.newaxis] * S
    
    acf_lags = acf[1:n]  # shape (n-1,)
    new_acf = delta + acf_lags[np.newaxis, :]
    new_psl = np.max(np.abs(new_acf), axis=1)
    
    best_i = int(np.argmin(new_psl))
    best_new_psl = new_psl[best_i]
    
    return best_i, float(best_new_psl), current_psl


def _fast_psl_optimize(h, max_passes=500, time_budget=20.0, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time.time() - start_time > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl


def _simulated_annealing_psl(h, time_budget=15.0, start_time=None):
    """
    Simulated annealing on PSL.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 5.0
    T_min = 0.1
    cooling = 0.97
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = np.random.randint(n)
        
        # Compute delta for this flip
        delta = np.zeros(n - 1)
        for k in range(1, n):
            contrib = 0.0
            if i - k >= 0:
                contrib += h[i - k]
            if i + k < n:
                contrib += h[i + k]
            delta[k - 1] = -2.0 * h[i] * contrib
        
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl


def _lfsr_init(n, seed=0):
    """Initialize using LFSR-based m-sequence."""
    # Use m=9 since 2^9-1=511 >= 400
    m = 9
    while (1 << m) - 1 < n:
        m += 1
    
    h = _generate_m_sequence(m, n, seed=seed)
    return h


def _random_init(n, seed=0):
    """Random ±1 initialization."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    return h


def _structured_init(n, seed=0):
    """
    Create structured initial sequences with better autocorrelation properties.
    """
    rng = np.random.default_rng(seed)
    
    # Option 1: Random
    h1 = rng.choice([-1.0, 1.0], size=n)
    
    # Option 2: LFSR-based
    h2 = _lfsr_init(n, seed=seed)
    
    # Option 3: Alternating with random perturbations
    h3 = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    flip_mask = rng.random(n) < 0.1
    h3[flip_mask] *= -1
    
    # Option 4: Based on sine wave
    freq = rng.random() * 5 + 1
    phase = rng.random() * 2 * np.pi
    h4 = np.sign(np.sin(2 * np.pi * freq * np.arange(n) / n + phase))
    h4 = np.where(h4 == 0, 1.0, h4)
    
    return [h1, h2, h3, h4]


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy:
    1. Try multiple structured and random initializations including LFSR-based m-sequences.
    2. For each, run fast gradient-based PSL optimization (Gordon's method).
    3. Use simulated annealing to escape local minima.
    4. Return the best sequence found.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin: 150s limit, leave 5s
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Fast gradient optimization from multiple diverse starts
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline - 30:  # Leave time for SA
            break
        
        # Get initial candidates
        candidates = _structured_init(N, seed=s * 7919 + 13)
        
        for h_init in candidates:
            if time.time() > deadline - 30:
                break
            
            # Run fast gradient optimization
            h_opt, psl_opt = _fast_psl_optimize(
                h_init, 
                max_passes=300,
                time_budget=8.0,
                start_time=start_time
            )
            
            if psl_opt < best_score:
                best_score = psl_opt
                best_h = h_opt.copy()
    
    # Phase 2: Simulated annealing from the best found so far
    if best_h is not None:
        if time.time() < deadline - 5:
            h_sa, psl_sa = _simulated_annealing_psl(
                best_h,
                time_budget=min(15.0, deadline - 5 - (time.time() - start_time)),
                start_time=start_time
            )
            
            if psl_sa < best_score:
                best_score = psl_sa
                best_h = h_sa
    
    # Phase 3: Final polish with gradient steps if time permits
    if best_h is not None and time.time() < deadline - 2:
        remaining_time = deadline - 2 - (time.time() - start_time)
        if remaining_time > 1:
            h_final, psl_final = _fast_psl_optimize(
                best_h,
                max_passes=100,
                time_budget=remaining_time,
                start_time=start_time
            )
            if psl_final < best_score:
                best_score = psl_final
                best_h = h_final
    
    # Ensure we have a valid result
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
        best_score = _score(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
