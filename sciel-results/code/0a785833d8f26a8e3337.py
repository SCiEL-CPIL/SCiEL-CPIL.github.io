"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _autoconv_max_abs(h):
    c = fftconvolve(h, h, mode='full')
    return float(np.max(np.abs(c)))


def _score(h):
    s = float(np.sum(h))
    if s == 0.0:
        return float('inf')
    return _autoconv_max_abs(h) / (s * s)


def _lfsr_sequence(n, poly):
    """Generate an n-bit sequence from a linear feedback shift register.
    poly is an integer whose set bits (excluding the top bit) give the
    feedback taps. Returns a list of 0/1 of length n."""
    if n <= 0:
        return []
    deg = poly.bit_length() - 1
    # state is deg bits; bit k (0-indexed from LSB) is state bit k
    state = 1  # non-zero seed
    out = []
    for _ in range(n):
        out.append(state & 1)
        # feedback: parity of bits at tap positions
        fb = 0
        x = state
        p = poly
        while p:
            if p & 1:
                fb ^= (x & 1)
            x >>= 1
            p >>= 1
        state = (state >> 1) | (fb << (deg - 1))
        if state == 0:
            state = 1
    return out


def _msequence_candidates():
    """Yield candidate ±1 sequences from primitive polynomials near deg 8,9,10."""
    cands = []
    # Known primitive polynomials (degree, poly in binary as int, LSB = x^0)
    # Represented as int where bit i set means tap at x^i.
    polys = [
        # degree 8
        0b110000111,   # x^8+x^4+x^3+x^2+1
        0b100011011,   # x^8+x^5+x^4+x+1
        0b100110001,   # x^8+x^6+x^5+x+1
        # degree 9
        0b1000100011,  # x^9+x^4+x^3+x+1
        0b1010000011,  # x^9+x^7+x+1
        0b1001101011,  # x^9+x^6+x^5+x^2+x+1
        # degree 10
        0b10000100011, # x^10+x^3+x+1
        0b10010001011, # x^10+x^7+x^4+x^2+x+1
        0b10000001011, # x^10+x^3+x+1 (alt)
        0b11000000011, # x^10+x^9+x+1
    ]
    for poly in polys:
        deg = poly.bit_length() - 1
        L = (1 << deg) - 1  # period
        bits = _lfsr_sequence(L, poly)
        # map 0->-1, 1->+1
        seq = np.where(np.array(bits, dtype=np.int8) == 0, -1, 1).astype(np.float64)
        # We need exactly N elements; take first N (or wrap). Use a window of length N.
        if L >= N:
            cands.append(seq[:N].copy())
        else:
            # repeat to fill N
            reps = int(np.ceil(N / L))
            cands.append(np.tile(seq, reps)[:N].copy())
    return cands


def _local_search(h, budget):
    """Greedy sign-flip local search to reduce the score."""
    h = h.copy()
    best = _score(h)
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        # Evaluate flipping each index; pick the best single flip that improves.
        base_max = _autoconv_max_abs(h)
        base_sum = float(np.sum(h))
        base_score = base_max / (base_sum * base_sum)
        best_gain = 0.0
        best_idx = -1
        # For speed, precompute autoconv once per round
        c = fftconvolve(h, h, mode='full')
        for i in range(N):
            if h[i] >= 0:
                continue  # only consider flipping +1 -> -1 (symmetry)
            # flipping h[i] to -h[i]
            # new autoconv: c_new = c - 2*h[i]*(h * e_i convolved) ... complex; recompute cheaply
            old = h[i]
            h[i] = -old
            new_max = float(np.max(np.abs(fftconvolve(h, h, mode='full'))))
            new_sum = base_sum - 2.0 * old
            if new_sum == 0.0:
                h[i] = old
                continue
            new_score = new_max / (new_sum * new_sum)
            gain = base_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            h[i] = old
        if best_idx >= 0 and best_gain > 1e-12:
            h[best_idx] = -h[best_idx]
            best = _score(h)
            improved = True
    return h, best


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin under 150s

    best_h = None
    best_score = float('inf')

    # 1) Evaluate m-sequence (LFSR) candidates
    for seq in _msequence_candidates():
        if time.time() > deadline:
            break
        sc = _score(seq)
        if sc < best_score:
            best_score = sc
            best_h = seq.copy()

    # 2) A few random ±1 sequences as a fallback / diversity
    rng = np.random.default_rng(12345)
    for _ in range(40):
        if time.time() > deadline:
            break
        seq = rng.choice([-1.0, 1.0], size=N)
        sc = _score(seq)
        if sc < best_score:
            best_score = sc
            best_h = seq.copy()

    if best_h is None:
        # ultimate fallback: all ones
        return np.ones(N)

    # 3) Local search polish on the best candidate
    remaining = deadline - time.time()
    if remaining > 3.0:
        polished, sc = _local_search(best_h, remaining - 5.0)
        if sc < best_score:
            best_score = sc
            best_h = polished

    # Final safety: if we blew the budget, still return the best we have.
    if time.time() > deadline - 2.0:
        return best_h

    return best_h
# EVOLVE-BLOCK-END
