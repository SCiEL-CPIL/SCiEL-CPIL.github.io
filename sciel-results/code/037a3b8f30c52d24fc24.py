import numpy as np
import time

N = 600
_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 120s kill

def _time_left():
    return _DEADLINE - time.time()

def _autoconv(h):
    # circular autoconvolution via FFT; h assumed sum-normalized
    F = np.fft.rfft(h, n=2 * N)
    c = np.fft.irfft(F * np.conj(F), n=2 * N)
    return c[:N]

def _peak(h):
    return float(np.max(_autoconv(h)))

def _l1(h):
    return float(np.sum(_autoconv(h)))

def _score(h):
    # normalized max autoconvolution = max(conv)/(sum h)^2
    s = float(np.sum(h))
    if s <= 0:
        return 1e9
    return float(np.max(_autoconv(h))) / (s * s)

def _project_nonneg(h):
    h = np.asarray(h, dtype=float)
    h[h < 0] = 0.0
    s = h.sum()
    if s <= 0:
        return np.full(N, 1.0 / N)
    return h / s

def _ipf(h, iters=40, lr=0.1):
    # minimize L1 of autoconvolution via multiplicative (IPF-style) updates
    h = _project_nonneg(h)
    for _ in range(iters):
        if _time_left() < 1.0:
            break
        c = _autoconv(h)          # c[k] = sum_j h[j]h[k-j]
        # gradient of L1 w.r.t. h[j] = sum_k c[k]*d(c[k])/dh[j] = 2*c (symmetric)
        g = 2.0 * c
        # multiplicative gradient step on nonnegative simplex
        h = h * np.exp(-lr * (g - g.mean()))
        h = _project_nonneg(h)
    return h

def _grad_peak(h, iters=80, lr=0.05):
    # projected gradient descent on the peak (subgradient)
    h = _project_nonneg(h)
    for _ in range(iters):
        if _time_left() < 1.0:
            break
        c = _autoconv(h)
        # subgradient of max(c) w.r.t. h: 2*c at argmax lags
        idx = np.argmax(c)
        g = np.zeros(N)
        g[idx] = 1.0
        g = 2.0 * g  # d(conv[k])/dh = 2*delta
        h = _project_nonneg(h - lr * g)
    return h

def _init_gauss(sigma):
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.exp(-0.5 * (x / sigma) ** 2)
    return _project_nonneg(h)

def _init_plateau(frac):
    h = np.zeros(N)
    m = max(1, int(N * frac))
    h[(N - m) // 2:(N - m) // 2 + m] = 1.0
    return _project_nonneg(h)

def construct():
    best_h = np.full(N, 1.0 / N)
    best_val = _score(best_h)  # ~1/600

    inits = [
        _init_gauss(0.10),
        _init_gauss(0.06),
        _init_gauss(0.15),
        _init_plateau(0.5),
        _init_plateau(0.7),
    ]

    for h0 in inits:
        if _time_left() < 2.0:
            break
        # IPF (L1 objective) then peak gradient polish
        h1 = _ipf(h0, iters=30, lr=0.1)
        v = _score(h1)
        if v < best_val:
            best_val, best_h = v, h1
        h2 = _grad_peak(h1, iters=40, lr=0.03)
        v = _score(h2)
        if v < best_val:
            best_val, best_h = v, h2

    # final polish on the best
    hp = _grad_peak(best_h, iters=30, lr=0.02)
    if _score(hp) < best_val:
        best_h = hp

    # return raw nonnegative heights (scale invariant, but keep sum=N for clarity)
    out = np.asarray(best_h, dtype=float) * N
    return [float(x) for x in out]
