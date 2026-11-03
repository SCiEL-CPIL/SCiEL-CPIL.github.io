"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated overlap, Nesterov accelerated gradient on soft-max,
and coordinate-descent bit-flipping to exploit discrete structure."""
import time
import numpy as np

N = 100
_T0 = time.time()
_DEADLINE = _T0 + 142  # safety margin before 150s kill


def _overlap_values_fft(h):
    """Compute C_k for k=0..N-1 via FFT.
    C_k = (2/N)*(N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    cs = _overlap_values_fft(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """Project h onto {0 <= h_i <= 1, sum(h) = target_sum} via bisection on lambda."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = target_sum
    # Bisection on the Lagrange multiplier
    lo = -np.max(h) - 2.0
    hi = 1.0 - np.min(h) + 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
    proj = np.clip(h - 0.5 * (lo + hi), 0.0, 1.0)
    # Fix residual
    diff = target - np.sum(proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argsort(proj)[-max(1, int(abs(diff))):]
        else:
            idx = np.argsort(proj)[:max(1, int(abs(diff)))]
        proj[idx] += diff / len(idx)
        proj = np.clip(proj, 0.0, 1.0)
    return proj


def _soft_max_val_grad(h, beta):
    """Value and gradient of soft-max of C_k."""
    n = len(h)
    cs = _overlap_values_fft(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    exps = np.exp(shifted)
    sum_exp = np.sum(exps)
    w = exps / sum_exp
    val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    return val, grad


def _nesterov_smooth(h0, beta, max_iter=400, lr_init=0.02):
    """Nesterov accelerated projected gradient on soft-max."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _soft_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next, N / 2.0)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr *= 0.9
    return h, _score(h)


def _bit_flip_improve(h, max_iters=2000):
    """Coordinate descent: try flipping each h_i toward 0 or 1, keep if improves score."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = _project(h, N / 2.0)
    cur = _score(h)
    improved = True
    it = 0
    while improved and it < max_iters:
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        it += 1
        for i in range(N):
            # Try setting h[i] to 0
            old = h[i]
            h[i] = 0.0
            h = _project(h, N / 2.0)
            s = _score(h)
            if s < cur - 1e-12:
                cur = s
                improved = True
            else:
                h[i] = old
                h = _project(h, N / 2.0)
            # Try setting h[i] to 1
            h[i] = 1.0
            h = _project(h, N / 2.0)
            s = _score(h)
            if s < cur - 1e-12:
                cur = s
                improved = True
            else:
                h[i] = old
                h = _project(h, N / 2.0)
    return h, cur


def _random_binary(start_count=10, rng=None):
    """Generate random 0/1 vectors with sum = N/2."""
    if rng is None:
        rng = np.random.default_rng(12345)
    results = []
    for _ in range(start_count):
        idx = rng.choice(N, size=N // 2, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        results.append(h)
    return results


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    rng = np.random.default_rng(7)
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Diverse continuous starts
    i = np.arange(N)
    starts = []
    starts.append(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N))
    starts.append(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N))
    starts.append(0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N))
    for _ in range(4):
        starts.append(rng.uniform(0, 1, N))

    # Random binary starts
    for hb in _random_binary(8, rng):
        starts.append(hb)

    # Evaluate and refine each start
    for h0 in starts:
        if time.time() > _DEADLINE - 15:
            break
        h0 = _project(h0, N / 2.0)
        _update(h0)

        # Nesterov smooth descent
        h1, s1 = _nesterov_smooth(h0, beta=80.0, max_iter=300, lr_init=0.02)
        _update(h1)

        # Bit-flip polish
        h2, s2 = _bit_flip_improve(h1, max_iters=1500)
        _update(h2)

    # Final high-beta polish on best
    if time.time() < _DEADLINE - 10:
        h3, s3 = _nesterov_smooth(best_h, beta=200.0, max_iter=400, lr_init=0.008)
        _update(h3)
        h4, s4 = _bit_flip_improve(h3, max_iters=2000)
        _update(h4)

    best_h = _project(best_h, N / 2.0)
    return best_h
