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
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip_steepest(h, max_passes=10, deadline=None):
    """
    Steepest-descent local optimization: in each pass, evaluate all single-flip
    candidates and apply the best one (or stop if none improves).
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        best_i = -1
        best_new_val = best_val
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            h[i] = -h[i]  # revert for evaluation
            if new_val < best_new_val:
                best_new_val = new_val
                best_i = i
        if best_i >= 0:
            h[best_i] = -h[best_i]
            best_val = best_new_val
        else:
            break
    return h, best_val

def _local_sign_flip_first_improve(h, max_passes=5, deadline=None):
    """
    First-improvement local optimization: scan and flip immediately if it helps.
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
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _generate_lfsr_sequence(n_bits, poly, init_state, n_out):
    """
    Generate a signed sequence from an LFSR (m-sequence).
    poly: integer representing the feedback polynomial (bitmask, excluding x^n term)
    init_state: initial state (non-zero integer)
    n_out: number of output bits
    Returns array of +/-1.
    """
    seq = np.zeros(n_out)
    state = init_state
    # Determine the LFSR width from the polynomial
    width = poly.bit_length()
    for i in range(n_out):
        # Output bit: least significant bit of state
        bit = state & 1
        seq[i] = 1.0 if bit == 1 else -1.0
        # Feedback: XOR of selected bits
        feedback = 0
        for j in range(width):
            if poly & (1 << j):
                feedback ^= (state >> j) & 1
        state = (state >> 1) | (feedback << (width - 1))
    return seq

def _generate_lfsr_primes(n_out, seed_offset=0):
    """
    Try several LFSR configurations known to produce m-sequences or good autocorrelation.
    Returns a list of candidate sequences.
    """
    candidates = []
    # Common primitive polynomials for various widths
    # For width 8: x^8 + x^4 + x^3 + x^2 + 1 (poly = 0x1D = 29, but we use lower bits)
    # Actually, let's use known primitive polynomials
    configs = [
        # (width, poly_mask, init_state)
        # Width 8, primitive: x^8+x^4+x^3+x^2+1 -> feedback taps at positions 4,3,2,0 -> mask=0b11101=29
        (8, 0b11101, 0b10110001),
        (8, 0b11101, 0b01101011),
        (8, 0b11101, 0b11001101),
        # Width 9, primitive: x^9+x^5+1 -> taps at 5,0 -> mask=0b100001=33
        (9, 0b100001, 0b100110101),
        (9, 0b100001, 0b011011010),
        # Width 10, primitive: x^10+x^3+1 -> taps at 3,0 -> mask=0b1001=9
        (10, 0b1001, 0b1011010011),
        (10, 0b1001, 0b0110101101),
        # Width 11, primitive: x^11+x^2+1 -> taps at 2,0 -> mask=0b101=5
        (11, 0b101, 0b10110100111),
        (11, 0b101, 0b01101101001),
        # Width 12, primitive: x^12+x^6+1 -> taps at 6,0 -> mask=0b1000001=65
        (12, 0b1000001, 0b101101001101),
        (12, 0b1000001, 0b011011010011),
        # Width 13, primitive: x^13+x^4+1 -> taps at 4,0 -> mask=0b10001=17
        (13, 0b10001, 0b1011010011011),
        (13, 0b10001, 0b0110110100110),
    ]
    for ci, (width, poly, init) in enumerate(configs):
        # Generate more than N bits, take first N
        seq = _generate_lfsr_sequence(width, poly, init, N)
        # Also try the complement (flip all signs)
        seq_comp = -seq
        candidates.append(seq)
        candidates.append(seq_comp)
        # Try a shifted version
        shift = (ci + seed_offset) % width
        if shift > 0:
            seq_shifted = np.roll(seq, shift)
            candidates.append(seq_shifted)
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)

    # Phase 1: Try LFSR/m-sequence initializations with local optimization
    lfsr_candidates = _generate_lfsr_primes(N)
    for ci, h_init in enumerate(lfsr_candidates):
        if time.time() > deadline:
            break
        # Ensure sum is not too small
        s = np.sum(h_init)
        if abs(s) < N * 0.05:
            # Try complement
            h_init = -h_init
            s = np.sum(h_init)
            if abs(s) < N * 0.05:
                continue
        # Use steepest descent for better convergence
        h_opt, val_opt = _local_sign_flip_steepest(h_init, max_passes=8, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: Diverse random/structured initializations
    n_restarts = 150
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        if r % 6 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 6 == 1:
            k = (r // 6) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 6 == 2:
            x = np.arange(N)
            freq = 1.0 + (r // 6) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2 * N)))
            h[h == 0] = 1.0
        elif r % 6 == 3:
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 6 == 4:
            h = np.ones(N)
            n_neg = np.random.randint(0, N // 2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        else:
            # Quadratic phase (number-theoretic)
            x = np.arange(N)
            p = (r // 6) % 7 + 3  # odd primes
            phase = (x * x) % p
            h = np.where(phase < p // 2, 1.0, -1.0)

        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0

        # Alternate between steepest and first-improve based on time remaining
        remaining = deadline - time.time()
        if remaining > 30:
            h_opt, val_opt = _local_sign_flip_steepest(h, max_passes=5, deadline=deadline)
        else:
            h_opt, val_opt = _local_sign_flip_first_improve(h, max_passes=3, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 3: Final polish on the best solution found
    if time.time() < deadline - 5.0:
        h_polish, val_polish = _local_sign_flip_steepest(best_h, max_passes=15, deadline=deadline)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
        # One more first-improve pass
        if time.time() < deadline - 2.0:
            h_polish2, val_polish2 = _local_sign_flip_first_improve(best_h, max_passes=5, deadline=deadline)
            if val_polish2 < best_val:
                best_val = val_polish2
                best_h = h_polish2

    return best_h

# EVOLVE-BLOCK-END
