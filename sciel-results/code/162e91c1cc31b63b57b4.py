import numpy as np
import time

N = 400
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

# Precompute FFT size for autoconvolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=_FFT_SIZE)
    C = np.fft.irfft(F * F, n=_FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    """Compute max(|autoconv|) / (sum(f))^2. Lower is better."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return float('inf')
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _generate_msequence(m, poly_coeffs):
    """
    Generate an m-sequence of length 2^m - 1 using an LFSR.
    poly_coeffs is a list of exponents for the polynomial x^m + sum(x^e) + 1? 
    Usually defined by the feedback taps.
    Let's use the standard representation: 
    The feedback is determined by the primitive polynomial.
    For x^m + c_{m-1}x^{m-1} + ... + c_1 x + 1 = 0.
    The LFSR state update:
    new_bit = XOR of bits at positions corresponding to non-zero coefficients (excluding x^m and x^0? No, usually including the taps).
    Let's use a simple implementation based on the polynomial exponents.
    """
    # poly_coeffs should be the exponents of the terms in x^m + sum(x^e) + 1.
    # e.g. for x^9 + x^4 + 1, coeffs are [4]. (x^9 and 1 are implicit)
    # Taps are at positions m and e.
    # In a Galois or Fibonacci LFSR, we need to be careful.
    # Let's use Fibonacci LFSR.
    # State is m bits.
    # Output is the LSB or MSB?
    # Let's use a standard reference implementation.
    
    L = (1 << m) - 1
    seq = np.zeros(L, dtype=np.int8)
    
    # State: list of bits, index 0 is LSB?
    # Let's use numpy array for state.
    # Initialize state to 1 (non-zero)
    state = np.zeros(m, dtype=np.int8)
    state[0] = 1
    
    # Feedback taps: indices where the polynomial has 1s, excluding the leading term x^m.
    # The polynomial is x^m + sum(x^{t_i}) + 1.
    # The feedback bit is the XOR of state[t_i] for all taps t_i (including 0 for the +1 term? No, usually the constant term is 1).
    # If poly is x^m + x^4 + 1, taps are 4 and 0.
    # new_bit = state[4] XOR state[0]
    # Then shift left, insert new_bit at MSB? Or shift right?
    # Standard Fibonacci:
    # b_{n} = sum c_i b_{n-i}
    # If we store state as [b_{n-1}, ..., b_{n-m}], it's confusing.
    
    # Let's use a simpler bitwise approach for small m.
    # State is an integer.
    # If m <= 32, we can use Python integers.
    mask = (1 << m) - 1
    state = 1
    seq = []
    
    # Taps are the exponents of the polynomial terms other than x^m.
    # e.g. x^9 + x^4 + 1 -> taps = [4, 0]
    taps = poly_coeffs
    
    for _ in range(L):
        seq.append(state & 1)
        # Calculate feedback
        fb = 0
        for t in taps:
            fb ^= (state >> t) & 1
        # Shift and insert
        state = (state >> 1) | (fb << (m - 1))
        state &= mask
        
    return np.array(seq, dtype=np.int8) * 2 - 1 # Map 0->-1, 1->1

def construct():
    """
    Minimizes max(|autoconv(f)|) / (sum(f))^2 using multi-start local search
    with FFT-accelerated scoring and structured initializations (M-sequences).
    """
    deadline = _start_time + _TIME_LIMIT
    
    best_f = np.ones(N, dtype=np.int8)
    best_score = _score(best_f)
    
    # Helper: try a candidate and update best
    def try_candidate(f):
        nonlocal best_f, best_score
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()

    # Phase 0: M-sequence initializations
    # Primitive polynomials:
    # m=8: x^8 + x^4 + x^3 + x^2 + 1 -> taps [4,3,2,0]
    # m=9: x^9 + x^4 + 1 -> taps [4,0]
    # m=10: x^10 + x^7 + 1 -> taps [7,0]
    # m=11: x^11 + x^2 + 1 -> taps [2,0]
    
    polys = {
        8: [4, 3, 2, 0],
        9: [4, 0],
        10: [7, 0],
        11: [2, 0],
    }
    
    for m, taps in polys.items():
        if time.time() > deadline:
            break
        try:
            seq = _generate_msequence(m, taps)
            L = len(seq)
            # Try all cyclic shifts of the m-sequence, truncated to N
            # If L >= N, we can take N consecutive bits from the cyclic sequence.
            # If L < N, we might repeat or just skip if it's too short?
            # For N=400, L=255 (m=8) is too short to take 400 unique bits without repeating.
            # Repeating an m-sequence degrades autocorrelation significantly.
            # So we prioritize m where L >= N.
            if L >= N:
                for shift in range(L):
                    if time.time() > deadline:
                        break
                    # Take N bits starting from shift
                    # seq is length L. We want seq[shift : shift+N]
                    # Since it's cyclic, we can use np.roll
                    # But np.roll is O(N). L is ~500-1000. This is fast.
                    f_cand = np.roll(seq, -shift)
                    f_cand = f_cand[:N]
                    try_candidate(f_cand)
        except Exception as e:
            pass

    # Phase 1: Other structured initializations (keep from previous if they help, but m-sequences should be strong)
    # 1.1 Thue-Morse
    f_tm = np.zeros(N, dtype=np.int8)
    f_tm[0] = 1
    for i in range(1, N):
        n = i // 2
        r = i % 2
        f_tm[i] = f_tm[n] if r == 0 else -f_tm[n]
    try_candidate(f_tm)
    
    # 1.2 Alternating
    f_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    try_candidate(f_alt)

    # Phase 2: Local search from best candidate
    def local_search(f, max_passes=10, time_budget=10.0):
        f = f.copy().astype(np.float64)
        S = np.sum(f)
        if abs(S) < 1e-6:
            return f, float('inf')
        
        C = _autoconv_fft(f)
        score = np.max(np.abs(C)) / (S * S)
        
        start = time.time()
        
        for _pass in range(max_passes):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() > deadline:
                    break
                
                v = f[i]
                # Flip f[i]
                f[i] = -v
                new_S = S - 2 * v
                
                # Compute new autoconvolution incrementally
                C_new = C.copy()
                k_start = max(0, i)
                k_end = min(2 * N - 1, i + N)
                
                idx = np.arange(k_start, k_end)
                f_idx = idx - i
                
                # Mask out center where k = 2i
                center_mask = (idx == 2 * i)
                delta = -4 * v * f[f_idx]
                delta[center_mask] = 0
                
                C_new[idx] += delta
                
                if new_S == 0:
                    new_score = float('inf')
                else:
                    new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - 1e-10:
                    # Accept
                    S = new_S
                    C = C_new
                    score = new_score
                    improved = True
                else:
                    # Revert
                    f[i] = v
        
            if not improved:
                break
        
        return f, score
    
    # Run local search from the best candidate found so far
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = local_search(best_f.astype(np.float64), max_passes=20, time_budget=30.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)

    # Phase 3: Multi-start local search from random restarts (reduced count to save time for m-seq polish)
    rng = np.random.default_rng(42)
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Random start
            f_rand = rng.choice([-1, 1], size=N).astype(np.float64)
            f_opt, sc_opt = local_search(f_rand, max_passes=5, time_budget=2.0)
            
            if sc_opt < best_score:
                best_score = sc_opt
                best_f = f_opt.astype(np.int8)

    # Phase 4: Final local search from best
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = local_search(best_f.astype(np.float64), max_passes=30, time_budget=35.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)

    return best_f.astype(np.int8)

# EVOLVE-BLOCK-END
