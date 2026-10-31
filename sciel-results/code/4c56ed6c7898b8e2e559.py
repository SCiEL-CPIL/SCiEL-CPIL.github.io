"""Rewritten program: Adam continuous optimization + mended Golay + multi-prime Legendre + 2-opt."""
N = 400
NFFT = 512  # >= 2*N-1 = 799? No: 2*400-1=799. Need power of 2 >= 799 -> 1024.
NFFT = 1024

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    H = np.fft.rfft(h, NFFT)
    ac = np.fft.irfft(H * H, NFFT)
    return float(np.max(np.abs(ac)))


def _score(h):
    """Objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_and_grad(h):
    """Objective and analytic gradient for continuous h.
    f = M / S^2, M = max_k |ac_k|, S = sum h.
    dM/dh = 2*sign(ac_k*) * ac_{k*-i}
    df/dh = (dM*S^2 - M*2*S*1) / S^4 = (dM*S - 2*M)/S^3
    """
    H = np.fft.rfft(h, NFFT)
    ac = np.fft.irfft(H * H, NFFT)
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e10, np.zeros(N)
    M = float(np.max(np.abs(ac)))
    # index of max |ac|
    k = int(np.argmax(np.abs(ac)))
    # dM/dh_i = 2 * sign(ac_k) * ac_{(k-i) mod NFFT}
    sign_k = np.sign(ac[k])
    if sign_k == 0:
        sign_k = 1.0
    # ac_{k-i} for i in 0..N-1: ac[(k - i) mod NFFT]
    idx = (k - np.arange(N)) % NFFT
    dM = 2.0 * sign_k * ac[idx]
    # df/dh = (dM * S - 2*M) / S^3
    grad = (dM * s - 2.0 * M) / (s ** 3)
    val = M / (s * s)
    return val, grad


def _adam_optimize(h0, deadline, max_iters=4000, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer on the continuous objective, with gradient clipping."""
    h = h0.astype(np.float64).copy()
    m = np.zeros(N)
    v = np.zeros(N)
    t = 0
    best_val = _score(h)
    best_h = h.copy()
    for t in range(1, max_iters + 1):
        if time.time() > deadline:
            break
        val, grad = _score_and_grad(h)
        if not np.isfinite(val):
            break
        # gradient clip
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 1e3:
            grad = grad * (1e3 / gnorm)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        h = h - lr * mhat / (np.sqrt(vhat) + eps)
        # clip to [-1, 1]
        np.clip(h, -1.0, 1.0, out=h)
        # track best (use quantized score for fair comparison? No, track continuous val)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    return best_h, best_val


def _quantize(h):
    """Quantize continuous h to +/-1, ensuring nonzero sum."""
    hq = np.sign(h)
    hq[hq == 0] = 1.0
    s = float(np.sum(hq))
    if abs(s) < 1e-9:
        hq[0] = 1.0
    return hq


def _local_sign_flip(h, max_passes=5, deadline=None):
    """1-opt: flip single elements, keep if improves."""
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
            nv = _score(h)
            if nv < best_val:
                best_val = nv
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val


def _two_opt(h, max_passes=5, deadline=None):
    """2-opt: flip pairs of elements. Try structured pairs (nearby + random)."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # candidate pairs: adjacent, distance 2, 3, and random
        pairs = []
        for d in (1, 2, 3):
            for i in range(N - d):
                pairs.append((i, i + d))
        # add random pairs
        for _ in range(200):
            a = np.random.randint(N)
            b = np.random.randint(N)
            if a != b:
                pairs.append((a, b))
        for (i, j) in pairs:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            h[j] = -h[j]
            nv = _score(h)
            if nv < best_val:
                best_val = nv
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
        if not improved:
            break
    return h, best_val


def _is_prime(m):
    if m < 2:
        return False
    if m == 2:
        return True
    if m % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(m)) + 1, 2):
        if m % i == 0:
            return False
    return True


def _next_prime(n):
    p = max(2, n)
    while not _is_prime(p):
        p += 1
    return p


def _legendre_sequence(n, prime_offset=0):
    """Legendre symbol sequence using prime p >= n + prime_offset."""
    p = _next_prime(n + prime_offset)
    half = (p - 1) // 2
    h = np.zeros(n)
    for x in range(n):
        if x == 0:
            h[x] = 1.0
        else:
            val = pow(x, half, p)
            if val == 1:
                h[x] = 1.0
            elif val == p - 1:
                h[x] = -1.0
            else:
                h[x] = 1.0
    return h


def _mended_golay():
    """Mended Golay complementary pair of length 256 -> take one sequence, pad to N=400.
    Golay pair: A^2 + B^2 has flat autoconvolution (only at lag 0).
    Mending: replace the 0 in the cross-correlation with a sign to make a valid +/-1 sequence.
    Here we construct the standard Golay pair recursively and use one arm.
    """
    # Golay pair construction: start with (1, 1) and (1, -1)
    # Recursion: (A, B) -> (A, A), (A, -B) and (A, A), (A, -B) for complementary
    A = np.array([1.0])
    B = np.array([1.0])
    # Build to length 256 = 2^8
    for _ in range(8):
        A = np.concatenate([A, A])
        B = np.concatenate([A, -B])
    # A and B are length 256, complementary: aca + acb = 0 for all non-zero lags
    # Use A as the base sequence. Pad with +1 to reach N=400.
    h = np.zeros(N)
    h[:256] = A
    h[256:] = 1.0
    return h


def _mended_golay_B():
    """Use the B arm of the Golay pair instead."""
    A = np.array([1.0])
    B = np.array([1.0])
    for _ in range(8):
        A = np.concatenate([A, A])
        B = np.concatenate([A, -B])
    h = np.zeros(N)
    h[:256] = B
    h[256:] = 1.0
    return h


def _chirp_sequence(n, k=1):
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * k * x * (x + 1) / (2 * n)))
    h[h == 0] = 1.0
    return h


def _block_pattern(seed):
    rng = np.random.RandomState(seed)
    h = np.ones(N)
    pos = 0
    while pos < N:
        block_len = rng.randint(4, 16)
        sign = rng.choice([-1.0, 1.0])
        h[pos:pos + block_len] = sign
        pos += block_len
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    np.random.seed(42)

    best_h = np.ones(N)
    best_val = _score(best_h)

    def consider(h):
        nonlocal best_h, best_val
        if time.time() > deadline:
            return
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # ---- Phase 1: algebraic basins with local refinement ----
    # Multi-prime Legendre ensemble
    for off in (0, 1, 2, 3, 5, 7, 11, 13):
        if time.time() > deadline:
            break
        h = _legendre_sequence(N, prime_offset=off)
        h, v = _local_sign_flip(h, max_passes=8, deadline=deadline)
        consider(h)
        h, v = _two_opt(h, max_passes=3, deadline=deadline)
        consider(h)

    # Mended Golay
    for gen in (_mended_golay, _mended_golay_B):
        if time.time() > deadline:
            break
        h = gen()
        h, v = _local_sign_flip(h, max_passes=8, deadline=deadline)
        consider(h)
        h, v = _two_opt(h, max_passes=3, deadline=deadline)
        consider(h)

    # Chirp
    for k in (1, 2, 3, 0.5):
        if time.time() > deadline:
            break
        h = _chirp_sequence(N, k=k)
        h, v = _local_sign_flip(h, max_passes=8, deadline=deadline)
        consider(h)

    # Block patterns
    for seed in (1, 2, 3, 4):
        if time.time() > deadline:
            break
        h = _block_pattern(seed)
        h, v = _local_sign_flip(h, max_passes=5, deadline=deadline)
        consider(h)

    # ---- Phase 2: Adam continuous optimization from diverse starts ----
    if time.time() < deadline - 20:
        starts = [
            _legendre_sequence(N, 0),
            _legendre_sequence(N, 3),
            _mended_golay(),
            _mended_golay_B(),
            _chirp_sequence(N, 1),
            np.random.choice([-1.0, 1.0], size=N),
            np.random.uniform(-1, 1, size=N),
        ]
        for s in starts:
            if time.time() > deadline:
                break
            # randomize slightly to escape saddle points
            h0 = s.astype(np.float64).copy()
            h0 += np.random.normal(0, 0.3, size=N)
            np.clip(h0, -1.0, 1.0, out=h0)
            h_cont, _ = _adam_optimize(h0, deadline, max_iters=3000, lr=0.02)
            hq = _quantize(h_cont)
            hq, v = _local_sign_flip(hq, max_passes=6, deadline=deadline)
            consider(hq)
            hq, v = _two_opt(hq, max_passes=3, deadline=deadline)
            consider(hq)

    # ---- Phase 3: refine best with more passes ----
    if time.time() < deadline - 5:
        h = best_h.copy()
        h, v = _local_sign_flip(h, max_passes=15, deadline=deadline)
        consider(h)
        h, v = _two_opt(h, max_passes=5, deadline=deadline)
        consider(h)
        h, v = _local_sign_flip(h, max_passes=10, deadline=deadline)
        consider(h)

    # ---- Phase 4: discrete random restarts if time remains ----
    if time.time() < deadline - 3:
        for r in range(30):
            if time.time() > deadline:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            # perturb from best
            if r % 2 == 0:
                flip_idx = np.random.choice(N, 5, replace=False)
                h[flip_idx] = -h[flip_idx]
            h, v = _local_sign_flip(h, max_passes=4, deadline=deadline)
            consider(h)

    return best_h

# EVOLVE-BLOCK-END
