"""Suite task: signed step heights minimizing max |autoconv| / integral^2."""
import time
import numpy as np

N = 400
DEADLINE = 115.0


def _autoconv(x):
    """Full linear autocorrelation (lag -N+1 .. N-1) via FFT."""
    n = N
    m = 2 * n - 1
    size = 1
    while size < m:
        size <<= 1
    F = np.fft.rfft(x, size)
    ac = np.fft.irfft(F * np.conj(F), size)[:m]
    # center so index 0 is lag 0
    return ac[n - 1:] + ac[n - 2::-1]


def _score(x):
    """Objective: max |autoconv| / (sum x)^2  (lower is better)."""
    ac = _autoconv(x)
    return float(np.max(np.abs(ac))) / (float(np.sum(x)) ** 2)


def _delta_ac(i, s):
    """Change to the full autoconvolution when x[i] is multiplied by s in [-1,1]."""
    n = N
    m = 2 * n - 1
    i = int(i)
    c = np.zeros(m)
    c[i] = s
    c[n - 1 - i] = s
    return np.fft.irfft(np.fft.rfft(c, 2 * m) * np.conj(np.fft.rfft(c, 2 * m)), 2 * m)[:m]


def _barker(n):
    """Best Barker sequence (length 7) tiled to n, with a random start offset."""
    base = np.array([1, 1, 1, -1, 1, 1, 1], dtype=np.int8)
    off = np.random.randint(0, 7)
    rep = np.tile(base, n // 7 + 1)[:n]
    return np.roll(rep, off)


def _refine(x, budget, seed):
    """Local sign-flip search using incremental autoconvolution updates."""
    rng = np.random.default_rng(seed)
    n = N
    m = 2 * n - 1
    x = x.astype(np.int8).copy()
    ac = _autoconv(x)
    cur = float(np.max(np.abs(ac)))
    # precompute delta autoconvolution for every position
    deltas = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        deltas[i] = _delta_ac(i, 2.0)
    order = rng.permutation(n)
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        for i in order:
            cand = ac - deltas[i]
            val = float(np.max(np.abs(cand)))
            if val + 1e-12 < cur:
                x[i] = -x[i]
                ac = cand
                cur = val
                improved = True
    return x, cur


def construct():
    t_start = time.time()
    rng = np.random.default_rng(0)

    # candidate pool of strong initial patterns
    base_barker = _barker(N)
    pool = []
    # Barker + random start offsets
    for off in range(0, 7):
        pool.append(np.roll(base_barker, off))
    # Barker + random sign flips (few)
    for k in range(8):
        x = base_barker.copy()
        idx = rng.choice(N, size=rng.integers(1, 6), replace=False)
        x[idx] = -x[idx]
        pool.append(x)
    # Barker + random block flips
    for k in range(8):
        x = base_barker.copy()
        start = int(rng.integers(0, N - 20))
        length = int(rng.integers(5, 40))
        x[start:start + length] = -x[start:start + length]
        pool.append(x)
    # random signs
    pool.append(np.where(rng.random(N) < 0.5, 1, -1).astype(np.int8))
    # random block pattern
    x = np.ones(N, dtype=np.int8)
    pos = 0
    while pos < N:
        length = int(rng.integers(10, 60))
        x[pos:pos + length] = 1 if (pos // 10) % 2 == 0 else -1
        pos += length
    pool.append(x)

    best = None
    best_val = np.inf
    # per-candidate time budget shrinks as we go
    per = 6.0
    for idx, x0 in enumerate(pool):
        if (time.time() - t_start) > DEADLINE - 5.0:
            break
        x, cur = _refine(x0, per, seed=idx)
        val = _score(x)
        if val < best_val:
            best_val = val
            best = x
    if best is None:
        best = _refine(base_barker, 4.0, seed=100)[0]

    # final polish pass with whatever time remains
    if (time.time() - t_start) < DEADLINE - 3.0:
        x, _ = _refine(best, max(1.0, DEADLINE - 3.0 - (time.time() - t_start)), seed=999)
        if _score(x) < best_val:
            best = x

    return best
