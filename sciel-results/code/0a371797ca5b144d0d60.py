"""Third autocorrelation inequality - signed step function search.

Minimize an autoconvolution-based upper bound on C3 over signed step
functions of length 400. Uses Thue-Morse / Legendre initialization plus
fast single-bit local search with incremental autoconvolution updates.
"""
import time
import numpy as np

N = 400
# Wall-clock budget for the evaluator is 150s; keep a safety margin.
_DEADLINE = time.time() + 140.0


def _autoconv(h):
    """Full linear autoconvolution (all lags) of a 1D sequence via FFT."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    f = np.fft.rfft(h.astype(np.float64), size)
    c = np.fft.irfft(f * f, size)[:2 * n - 1]
    return c


def _autoconv_direct(h):
    """Direct O(N^2) autoconvolution (reference / for verification)."""
    n = len(h)
    c = np.zeros(2 * n - 1)
    for k in range(-(n - 1), n):
        if k >= 0:
            c[n - 1 + k] = np.dot(h[:n - k], h[k:])
        else:
            c[n - 1 + k] = np.dot(h[-k:], h[:n + k])
    return c


def _score(h):
    """Upper bound on C3: max absolute value of the autoconvolution."""
    c = _autoconv(h)
    return float(np.max(np.abs(c)))


def _thue_morse(n):
    """Thue-Morse sequence mapped to {+1, -1}."""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        bits[i] = bits[i >> 1] ^ (i & 1)
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1


def _legendre(n):
    """Legendre-type (quadratic character mod p) signed sequence."""
    # find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        i = 2
        while i * i <= m:
            if m % i == 0:
                return False
            i += 1
        return True
    p = n
    while not is_prime(p):
        p += 1
    out = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        # Legendre symbol (i/p): 1 if quadratic residue, -1 otherwise
        out[i] = 1 if pow(i, (p - 1) // 2, p) == 1 else -1
    return out


def _rademacher_product(n, k):
    """Product of k Rademacher (Walsh) sequences - low discrepancy."""
    out = np.ones(n, dtype=np.int8)
    for j in range(k):
        period = 1 << (j + 1)
        r = np.ones(n, dtype=np.int8)
        for i in range(n):
            if (i >> j) & 1:
                r[i] = -1
        out = out * r
    return out


def _autoconv_init(h):
    """Initialize incremental autoconvolution state."""
    return _autoconv_direct(h).copy()


def _flip_update(c, h, i, s):
    """Return new autoconvolution after flipping bit i (currently s) to -s.

    c[k] = sum_j h[j] h[j+k]. Flipping h[i]: h[i] -> -h[i].
    Terms involving h[i] change sign. For lag d = k - (n-1):
      c[d] += -2 s * (h[i+d] + h[i-d])   (with in-range guards)
    We update c in place and return it.
    """
    n = len(h)
    # lags d from -(n-1) .. (n-1); index = n-1+d
    for d in range(-(n - 1), n):
        j1 = i + d
        j2 = i - d
        contrib = 0.0
        if 0 <= j1 < n:
            contrib += h[j1]
        if 0 <= j2 < n and d != 0:
            # avoid double counting the i term when d==0 (j1==j2==i)
            contrib += h[j2]
        c[n - 1 + d] += -2.0 * s * contrib
    # apply the flip to h in place
    h[i] = -s
    return c


def _max_abs(c):
    return float(np.max(np.abs(c)))


def _local_search(h0, time_budget, seed=0):
    """Fast single-bit local search (greedy + light annealing).

    Returns (best_h, best_score).
    """
    rng = np.random.default_rng(seed)
    h = h0.astype(np.int8).copy()
    c = _autoconv_direct(h).copy()
    cur = _max_abs(c)
    best_h = h.copy()
    best = cur

    n = len(h)
    # Precompute index list for random sampling
    idx = np.arange(n)

    # Phase 1: greedy single-bit descent (fast, deterministic-ish)
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        # scan in a shuffled order to break ties
        order = rng.permutation(n)
        for i in order:
            s = h[i]
            c2 = c.copy()
            _flip_update(c2, h, i, s)  # this flips h in place!
            # undo the flip in h (we only want to test)
            h[i] = s
            val = _max_abs(c2)
            if val < cur - 1e-12:
                # accept: re-apply flip to c and h
                _flip_update(c, h, i, s)
                cur = val
                if cur < best - 1e-12:
                    best = cur
                    best_h = h.copy()
                improved = True
                break  # restart scan after an improvement

    # Phase 2: simulated annealing with single-bit moves
    if time.time() < time_budget:
        T = 0.5 * cur
        T_min = 1e-6
        cool = 0.9995
        iters = 0
        while T > T_min and time.time() < time_budget:
            i = int(rng.integers(0, n))
            s = h[i]
            c2 = c.copy()
            _flip_update(c2, h, i, s)
            h[i] = s
            val = _max_abs(c2)
            d = val - cur
            if d < 0 or rng.random() < np.exp(-d / T):
                _flip_update(c, h, i, s)
                cur = val
                if cur < best - 1e-12:
                    best = cur
                    best_h = h.copy()
            T *= cool
            iters += 1
            if iters % 20000 == 0 and time.time() >= time_budget:
                break

    return best_h, best


def construct():
    """Construct a signed step function minimizing the C3 autoconvolution bound."""
    t0 = time.time()
    budget = _DEADLINE - t0
    if budget < 5.0:
        # Not enough time; return a strong deterministic seed.
        return _thue_morse(N).tolist()

    # Candidate initializations (deterministic, low-discrepancy).
    inits = [
        _thue_morse(N),
        _legendre(N),
        _rademacher_product(N, 3),
        _rademacher_product(N, 5),
        -_thue_morse(N),
    ]

    # Score each init, keep the best few as seeds for local search.
    scored = []
    for h in inits:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])

    best_h, best = scored[0][1], scored[0][0]

    # Allocate remaining time across a few local-search restarts.
    n_restarts = 4
    per = (budget - 5.0) / n_restarts
    for r in range(n_restarts):
        seed_h = scored[r % len(scored)][1]
        # add a small deterministic perturbation for restart diversity
        if r > 0:
            rng = np.random.default_rng(1000 + r)
            seed_h = seed_h.astype(np.int8).copy()
            k = rng.integers(1, 20)
            flips = rng.choice(N, size=k, replace=False)
            seed_h[flips] *= -1
        t_end = t0 + 5.0 + (r + 1) * per
        if time.time() >= t_end:
            break
        h, s = _local_search(seed_h, t_end, seed=7 + r)
        if s < best - 1e-12:
            best = s
            best_h = h

    # Final high-precision polish: a few greedy passes on the best.
    if time.time() < _DEADLINE - 3.0:
        h, s = _local_search(best_h, _DEADLINE - 3.0, seed=999)
        if s < best - 1e-12:
            best = s
            best_h = h

    return best_h.astype(int).tolist()
