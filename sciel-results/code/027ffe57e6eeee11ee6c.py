"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Strategy:
1. Use FFT-based autoconvolution for fast scoring.
2. Start from multiple diverse initializations (random, structured patterns).
3. Perform simulated annealing with single-bit flips, tracking the best solution.
4. Use a deadline guard to return the best solution before timeout.
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _autoconv_direct(f):
    """Compute linear autoconvolution directly (slower but exact for small N)."""
    n = len(f)
    ac = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        # ac[k] = sum_i f[i] * f[k-i] where indices in [0, n-1]
        i_lo = max(0, k - (n - 1))
        i_hi = min(n - 1, k)
        if i_lo <= i_hi:
            ac[k] = np.dot(f[i_lo:i_hi + 1], f[k - i_hi:k - i_lo + 1][::-1])
    return ac


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    rng = np.random.RandomState(12345)

    # Generate diverse initial candidates
    candidates = []

    # Random sequences
    for seed in range(10):
        r = np.random.RandomState(seed * 1000 + 42)
        candidates.append(r.choice([-1.0, 1.0], size=N))

    # Alternating pattern
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))

    # Block patterns
    for block_size in [2, 4, 8, 16, 32, 64]:
        n_blocks = (N + block_size - 1) // block_size
        pattern = []
        for b in range(n_blocks):
            sign = 1.0 if b % 2 == 0 else -1.0
            pattern.extend([sign] * block_size)
        candidates.append(np.array(pattern[:N]))

    # Golay-like construction
    def golay_pair(n):
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2

    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])

    # Barker sequence tiling
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])

    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()

    f = best_f.copy()
    cur_score = best_score

    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.99998

    while _remaining() > 5.0 and T > T_min:
        # Try a random single-bit flip
        idx = rng.randint(N)
        old_val = f[idx]
        f[idx] = -f[idx]

        ac = _autoconv_fft(f)
        s = np.sum(f)
        if abs(s) < 1e-12:
            f[idx] = old_val
            T *= cooling
            continue

        new_score = np.max(np.abs(ac)) / (s * s)

        # Accept if better or with Metropolis probability
        if new_score < cur_score or rng.random() < np.exp(-(new_score - cur_score) / T):
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
        else:
            f[idx] = old_val

        T *= cooling

    # Final greedy pass
    f = best_f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return best_f
    cur_score = np.max(np.abs(ac)) / (s * s)

    improved = True
    while improved and _remaining() > 2.0:
        improved = False
        for idx in range(N):
            f[idx] = -f[idx]
            ac_new = _autoconv_fft(f)
            s_new = np.sum(f)
            if abs(s_new) < 1e-12:
                f[idx] = -f[idx]
                continue
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < cur_score:
                cur_score = new_score
                ac = ac_new
                s = s_new
                improved = True
            else:
                f[idx] = -f[idx]
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()

    return best_f
