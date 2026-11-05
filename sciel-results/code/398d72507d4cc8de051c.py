"""Fresh design: anti-periodic base + smooth-max gradient descent."""
N = 400
import time
import numpy as np

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

def _value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _smooth_value(h, beta):
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    m = np.max(ac)
    Z = np.sum(np.exp(beta * (ac - m)))
    sm = m + np.log(Z) / beta
    return 2.0 * N * sm / (s * s)

def _grad(h, beta):
    s = np.sum(h)
    ac = np.convolve(h, h)
    m = np.max(ac)
    w = np.exp(beta * (ac - m))
    Z = np.sum(w)
    p = w / Z
    sm = m + np.log(Z) / beta
    # d(sm)/d ac_j = p_j
    g = np.zeros(N)
    for j in range(len(ac)):
        c = p[j]
        if c == 0.0:
            continue
        i = j
        k = j - (N - 1)
        g[i] += 2.0 * c * h[k]
        g[k] += 2.0 * c * h[i]
    val = 2.0 * N * sm / (s * s)
    return val, g / (s * s)

def _project(h):
    # keep finite, finite sum
    if not np.all(np.isfinite(h)):
        h = np.nan_to_num(h, nan=0.0, posinf=1e3, neginf=-1e3)
    if abs(np.sum(h)) < 1e-8:
        h = h + 1.0
    return h

def _construct():
    # anti-periodic base: two identical blocks B, second negated
    half = N // 2
    B = np.ones(half)
    h = np.concatenate([B, -B])
    h = h + 1e-3 * (np.random.default_rng(0).normal(size=N))

    best_h = h.copy()
    best_v = _value(h)
    if not np.isfinite(best_v):
        best_h = np.ones(N)
        best_v = _value(best_h)

    beta = 20.0
    lr = 0.5
    t = _START
    while time.time() < _DEADLINE - _MARGIN:
        try:
            sv, g = _grad(h, beta)
            if not np.isfinite(sv):
                beta *= 0.5
                continue
            g = g - np.mean(g)          # invariance to constant shift (keeps sum)
            h = _project(h - lr * g)
            # adapt lr
            if sv < best_v:
                best_h = h.copy()
                best_v = sv
                lr *= 1.1
            else:
                lr *= 0.85
            beta = min(beta * 1.02, 400.0)
        except Exception:
            break

    # final local polish with high beta
    h = best_h.copy()
    beta = 300.0
    lr = 0.1
    t = _START
    while time.time() < _DEADLINE - _MARGIN:
        try:
            sv, g = _grad(h, beta)
            if not np.isfinite(sv):
                break
            g = g - np.mean(g)
            h = _project(h - lr * g)
            if sv < best_v:
                best_h = h.copy()
                best_v = sv
                lr *= 1.05
            else:
                lr *= 0.9
        except Exception:
            break

    # guarantee finite, non-zero sum, finite value
    best_h = _project(best_h)
    if not np.isfinite(_value(best_h)):
        best_h = np.ones(N)
    return best_h

def construct():
    try:
        return _construct()
    except Exception:
        return np.ones(N)
