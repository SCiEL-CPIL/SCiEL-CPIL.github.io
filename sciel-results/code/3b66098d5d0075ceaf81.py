"""Discrete spectral design: minimize autoconvolution peak at fixed L2 norm."""
import time
import math
import numpy as np

N = 50
DEADLINE = 140.0  # safety margin under 150s wall-clock

# Build the banded autoconvolution linear structure.
# Lags k = -(N-1) .. (N-1). For lag k, valid j: max(0,-k) <= j < min(N, N-k).
_LAGS = np.arange(-(N - 1), N)
_I = []
_J = []
for k in _LAGS:
    lo = max(0, -k)
    hi = min(N, N - k)
    if hi > lo:
        js = np.arange(lo, hi)
        _I.append(js)
        _J.append(js + k)
_I = np.concatenate(_I) if _I else np.zeros(0, dtype=int)
_J = np.concatenate(_J) if _J else np.zeros(0, dtype=int)
_M = len(_I)
_LAG_START = np.concatenate([np.cumsum([0] + [len(x) for x in _I]), [_M]])
_LAG_OF_M = np.zeros(_M, dtype=int)
for k in range(2 * N - 1):
    a, b = _LAG_START[k], _LAG_START[k + 1]
    _LAG_OF_M[a:b] = k


def _conv_full(h):
    """Autoconvolution over all 2N-1 lags."""
    prod = h[_I] * h[_J]
    out = np.empty(2 * N - 1)
    np.add.reduceat(prod, _LAG_START[:-1], out=out)
    return out


def _score(h):
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    peak = float(np.max(_conv_full(h)))
    if peak <= 0.0:
        return 0.0
    return l2sq / (l1 * peak)


def _smooth_peak(h, tau):
    conv = _conv_full(h)
    m = float(np.max(conv))
    if m <= 0.0:
        return 0.0, np.zeros_like(conv)
    e = np.exp((conv - m) / tau)
    s = float(np.sum(e))
    w = e / s
    val = m + tau * math.log(s)
    return val, w


def _grad_smooth_peak(h, w, tau):
    """Gradient of smooth-peak w.r.t. h."""
    g = np.zeros(N)
    # d conv[k]/d h[i] = sum_{m: I_m=i} h[J_m] + sum_{m: J_m=i} h[I_m]
    np.add.at(g, _I, w[_LAG_OF_M] * h[_J])
    np.add.at(g, _J, w[_LAG_OF_M] * h[_I])
    return g


def _construct_one(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = np.zeros(N)
    pos = rng.choice(N, size=N // 2, replace=False)
    h[pos] = 1.0
    eps = 1e-8
    x = np.log(h + eps)
    best_h = h.copy()
    best_s = _score(h)
    target_l2 = np.sqrt(float(np.dot(best_h, best_h)))
    tau = 1.0
    lr = 0.02
    t0 = time.time()
    while time.time() - t0 < time_budget:
        hcur = np.exp(x)
        val, w = _smooth_peak(hcur, tau)
        g = _grad_smooth_peak(hcur, w, tau)
        x = x - lr * g * hcur
        hcur = np.exp(x)
        nrm = math.sqrt(float(np.dot(hcur, hcur)))
        if nrm > 0:
            hcur = hcur * (target_l2 / nrm)
            x = np.log(hcur + eps)
        s = _score(hcur)
        if s > best_s:
            best_s = s
            best_h = hcur.copy()
        tau = max(0.05, tau * 0.97)
        lr = max(0.001, lr * 0.995)
    return best_h, best_s


def construct():
    t_start = time.time()
    best_h = np.ones(N)
    best_s = _score(best_h)
    seeds = [12345, 777, 20240517, 999999, 42, 31337, 555, 88888, 1, 2, 3]
    n_restarts = 14
    for i in range(n_restarts):
        if time.time() > DEADLINE:
            break
        remain = DEADLINE - time.time()
        if remain <= 5.0:
            break
        per = min(8.0, max(1.0, (DEADLINE - time.time()) / max(1, n_restarts - i)))
        s = seeds[i % len(seeds)] + i * 1000
        h, sc = _construct_one(s, per)
        if sc > best_s:
            best_s = sc
            best_h = h
    return list(best_h)
