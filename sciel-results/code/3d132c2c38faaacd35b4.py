"""Signed equal-width step functions minimizing max|autoconvolution| / (integral)^2.

Objective: score = max_k |c[k]| / (sum_j x[j])^2,  where c = x * x (autoconvolution).
Lower is better. We use signed (+/-1) step heights.

Strategy (algebraic structure + local refinement):
  1. Construct an m-sequence (maximum-length LFSR sequence) over GF(2^k) which has
     near-flat autocorrelation: sidelobes are exactly -1 for all non-zero lags.
     This gives a theoretically optimal peak-to-average ratio structure.
     We map GF(2) bits to +/-1 signs, and bias the sequence toward +1 to maximize
     the integral while preserving the low sidelobe structure.
  2. Use Legendre symbol sequences (quadratic residues) as an alternative algebraic
     construction with provably bounded off-peak autocorrelation.
  3. Refine with a fast greedy coordinate-descent local search that flips individual
     signs to reduce the objective, with an explicit wall-clock deadline guard.
  4. Combine with multiple diverse algebraic initializations and run local search
     on each, keeping the best result.
"""

import time
import numpy as np

N = 400
DEADLINE_S = 135.0


def _autoconv(x):
    """Full autoconvolution c[k] = sum_j x[j] * x[j+k] for k in [0, N-1].
    Uses FFT for O(N log N) cost."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    c = np.fft.irfft(f * np.conj(f), n=2 * n)
    return c[:n]


def _score(x):
    """max |autoconvolution| / (integral)^2. Returns inf if integral ~ 0."""
    I = float(np.sum(x))
    if abs(I) < 1e-9:
        return float("inf")
    c = _autoconv(x)
    m = float(np.max(np.abs(c)))
    return m / (I * I)


def _msequence(n, primitive_poly):
    """Generate an m-sequence of length n using a primitive polynomial.
    primitive_poly: list of exponents (including 0) of the primitive polynomial.
    Returns a sequence of +/-1 of length n."""
    # Find the degree k such that 2^k - 1 >= n
    k = 1
    while (1 << k) - 1 < n:
        k += 1
    
    # Use the given primitive polynomial for degree k
    # The polynomial is x^k + sum(x^i for i in primitive_poly)
    # LFSR: state is k bits, feedback = XOR of bits at positions in primitive_poly
    
    # Initialize state (non-zero)
    state = 1  # all 1s or just 1
    seq = []
    full_len = (1 << k) - 1
    
    for _ in range(full_len):
        # Output bit is the last bit of state
        bit = state & 1
        seq.append(1.0 if bit == 0 else -1.0)
        
        # Compute feedback
        feedback = 0
        for p in primitive_poly:
            if p < k:
                feedback ^= (state >> p) & 1
        # Shift right and insert feedback at top
        state = ((state >> 1) | (feedback << (k - 1))) & ((1 << k) - 1)
    
    # Truncate or pad to length n
    if len(seq) >= n:
        return seq[:n]
    else:
        # Pad with +1
        return seq + [1.0] * (n - len(seq))


def _legendre_sequence(n, p):
    """Generate a Legendre symbol sequence of length n.
    For each position i, compute the Legendre symbol (i/p) mod p,
    mapping 0 -> +1, 1 -> +1, -1 -> -1.
    This gives bounded off-peak autocorrelation."""
    seq = np.ones(n)
    for i in range(1, n):
        # Compute Legendre symbol (i/p)
        val = pow(i % p, (p - 1) // 2, p)
        if val == 0:
            seq[i] = 1.0  # i is multiple of p
        elif val == 1:
            seq[i] = 1.0  # quadratic residue
        else:
            seq[i] = -1.0  # non-residue
    return seq


def _greedy_local_search(x, deadline):
    """Greedy coordinate descent: repeatedly flip the sign of the index whose
    flip most reduces the objective. Stops on no-improvement sweep or deadline."""
    x = x.copy()
    cur = _score(x)
    if not np.isfinite(cur):
        return x, cur
    n = len(x)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_i = -1
        best_val = cur
        # Use random order to avoid bias
        order = np.random.permutation(n)
        for i in order:
            if time.time() > deadline:
                break
            x[i] = -x[i]
            v = _score(x)
            x[i] = -x[i]
            if v < best_val - 1e-12:
                best_val = v
                best_i = i
        if best_i >= 0:
            x[best_i] = -x[best_i]
            cur = best_val
            improved = True
    return x, cur


def _biased_msequence(n, primitive_poly, bias_ratio):
    """Create a biased m-sequence: start with m-sequence, then flip some -1s to +1
    to increase the integral while preserving structure."""
    base = _msequence(n, primitive_poly)
    x = np.array(base)
    # Find positions where x is -1
    neg_idx = np.where(x == -1.0)[0]
    if len(neg_idx) > 0:
        # Flip a fraction of them to +1
        n_flip = int(len(neg_idx) * bias_ratio)
        if n_flip > 0:
            flip_idx = np.random.choice(neg_idx, n_flip, replace=False)
            x[flip_idx] = 1.0
    return x


def _diverse_initializations(rng):
    """Generate diverse algebraic initial sign sequences."""
    inits = []
    
    # 1. M-sequences with different primitive polynomials
    # Primitive polynomials for degrees k where 2^k-1 >= 400, so k >= 9
    # x^9 + x^4 + 1 (primitive)
    inits.append(_biased_msequence(N, [4, 0], 0.0))
    # x^9 + x^5 + 1 (primitive)  
    inits.append(_biased_msequence(N, [5, 0], 0.0))
    # x^10 + x^3 + 1 (primitive)
    inits.append(_biased_msequence(N, [3, 0], 0.0))
    # x^11 + x^2 + 1 (primitive)
    inits.append(_biased_msequence(N, [2, 0], 0.0))
    
    # 2. Biased m-sequences (flip some negatives to positives)
    inits.append(_biased_msequence(N, [4, 0], 0.3))
    inits.append(_biased_msequence(N, [5, 0], 0.3))
    inits.append(_biased_msequence(N, [3, 0], 0.5))
    
    # 3. Legendre symbol sequences with different primes
    for p in [401, 409, 419, 431, 433, 439, 443, 449]:
        if p > N:
            inits.append(_legendre_sequence(N, p))
    
    # 4. Structured: mostly +1 with periodic negatives
    x = np.ones(N, dtype=np.float64)
    x[::4] = -1.0
    inits.append(x)
    
    # 5. Random +/-1
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 6. Low frequency cosine patterns
    for k in [1, 2, 3, 5, 7]:
        t = np.arange(N)
        h = np.sign(np.cos(2 * np.pi * k * t / N))
        h[h == 0] = 1.0
        inits.append(h)
    
    # 7. Block structures
    for _ in range(3):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.randint(3, 15)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos + block_len] = sign
            pos += block_len
        inits.append(h)
    
    # 8. Mostly +1 with sparse random negatives
    for n_neg in [N // 8, N // 6, N // 5]:
        h = np.ones(N)
        neg_idx = rng.choice(N, n_neg, replace=False)
        h[neg_idx] = -1.0
        inits.append(h)
    
    # 9. Periodic patterns with different periods
    for period in [3, 5, 7, 11]:
        pattern = []
        for i in range(period):
            if i < period - 1:
                pattern.append(1.0)
            else:
                pattern.append(-1.0)
        h = np.tile(pattern, N // period + 1)[:N]
        inits.append(h)
    
    return inits


def construct():
    start = time.time()
    deadline = start + DEADLINE_S
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = _diverse_initializations(rng)
    
    # Sort by initial score to prioritize promising starts
    scored_inits = []
    for x in inits:
        s = _score(x)
        if np.isfinite(s):
            scored_inits.append((s, x))
    
    scored_inits.sort(key=lambda t: t[0])
    
    best_x = np.ones(N, dtype=np.float64)
    best_val = _score(best_x)
    
    # Refine each initialization with local search, keeping the best
    for init_score, x in scored_inits:
        if time.time() > deadline:
            break
        
        x_opt, val_opt = _greedy_local_search(x, deadline)
        
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Safety: ensure a finite, valid result is always returned.
    if not np.isfinite(best_val):
        best_x = np.ones(N, dtype=np.float64)
        best_val = _score(best_x)
    
    return best_x


if __name__ == "__main__":
    res = construct()
    print("score:", _score(res))
