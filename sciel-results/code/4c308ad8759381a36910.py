"""Reframe: minimize max over ALL lags of autoconvolution via softmax surrogate + FFT."""
import time
import numpy as np

N = 600
DX = 1.0 / (2.0 * N)  # interval width on [-1/4, 1/4]

_DEADLINE = time.time() + 135.0  # safety margin under 150s kill


def _autoconv(h):
    """Full linear autoconvolution of step heights, scaled by DX (dx cancels in ratio)."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    m = 2 * n - 1
    # use next power of 2 for FFT speed
    L = 1
    while L < m:
        L <<= 1
    F = np.fft.rfft(h, n=L)
    A = np.fft.irfft(F * F.conj(), n=L)[:m]
    return A


def _score(h):
    """Normalized maximum autoconvolution (scale-invariant)."""
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0:
        return float("inf")
    A = _autoconv(h)
    peak = float(A.max())
    # normalize: divide by (integral)^2 / dx ; integral = s*DX, so norm = s^2*DX
    # ratio = peak / (s^2 * DX)
    return peak / (s * s * DX)


def _grad(h):
    """Gradient of peak autoconvolution w.r.t. h (subgradient at argmax lag)."""
    A = _autoconv(h)
    k = int(np.argmax(A))
    # conv[k] = sum_i h[i] h[i+k-?]; for linear conv A[k]=sum_j h[j]*h[j+k] (k>=0)
    # dA[k]/dh[j] = h[j+k] (for j in range where j+k < n) + h[j-k] (for j-k>=0)
    n = len(h)
    g = np.zeros(n)
    if k == 0:
        g[:] = 2.0 * h
    else:
        # j such that j+k < n: contribution h[j+k]
        g[:n - k] += h[k:]
        # j such that j-k >= 0: contribution h[j-k]
        g[k:] += h[:n - k]
    return g


def _proj_simplex(v, target):
    """Closest point (Euclidean) to v on {x>=0, sum x = target}."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (sv - target))[0]
    if len(rho) == 0:
        return np.zeros(n)
    r = rho[-1]
    theta = (sv[r] - target) / (r + 1.0)
    return np.maximum(v - theta, 0.0)


def _softmax_obj(h, tau):
    """Smooth surrogate: soft-max of autoconvolution peaks / (sum^2 DX)."""
    A = _autoconv(h)
    s = h.sum()
    if s <= 0:
        return float("inf")
    z = A / tau
    z = z - z.max()
    m = np.exp(z).sum()
    softmax_val = tau * np.log(m) + tau * z.max()
    return softmax_val / (s * s * DX)


def _softmax_grad(h, tau):
    A = _autoconv(h)
    s = h.sum()
    if s <= 0:
        return np.zeros(N)
    z = A / tau
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    # d softmax / dA[k] = w[k]
    # dA/dh = sum_k w[k] dA[k]/dh
    g = np.zeros(N)
    n = len(h)
    for k in range(len(A)):
        wk = w[k]
        if wk < 1e-12:
            continue
        if k == 0:
            g[:] += wk * 2.0 * h
        else:
            g[:n - k] += wk * h[k:]
            g[k:] += wk * h[:n - k]
    # divide by s^2 DX
    return g / (s * s * DX)


def _line_search(h, g, f0, max_step=2.0):
    """Backtracking line search on the softmax objective with projection."""
    # direction: -g, projected step
    ts = [max_step * (0.5 ** i) for i in range(12)]
    best_h, best_f = h.copy(), f0
    for t in ts:
        cand = _proj_simplex(h - t * g, h.sum())
        f = _softmax_obj(cand, TAU)
        if f < best_f:
            best_h, best_f = cand, f
        else:
            break
    return best_h, best_f


def _init_candidates():
    cands = []
    x = np.linspace(-1, 1, N)
    # uniform
    cands.append(np.ones(N))
    # flat plateau of various widths
    for wf in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w = int(N * wf)
        v = np.zeros(N)
        v[(N - w) // 2:(N + w) // 2] = 1.0
        cands.append(v)
    # smooth raised-cosine (low spectral peak)
    for wf in [0.6, 0.8, 1.0]:
        w = int(N * wf)
        v = np.zeros(N)
        idx = np.arange(w)
        seg = 0.5 * (1 + np.cos(2 * np.pi * idx / max(w, 1)))
        v[(N - w) // 2:(N + w) // 2] = seg
        cands.append(v)
    # Gaussian
    for sig in [0.4, 0.6, 0.8, 1.0]:
        cands.append(np.exp(-0.5 * (x / sig) ** 2))
    # two separated bumps
    for sep in [0.25, 0.4]:
        v = np.zeros(N)
        bw = int(N * 0.25)
        c1 = int(N * (0.5 - sep))
        c2 = int(N * (0.5 + sep))
        v[max(0, c1 - bw // 2):min(N, c1 + bw // 2)] = 1.0
        v[max(0, c2 - bw // 2):min(N, c2 + bw // 2)] = 1.0
        cands.append(v)
    # random smooth
    rng = np.random.default_rng(7)
    for _ in range(4):
        r = rng.standard_normal(N)
        # low-pass: convolve with small box
        r = np.convolve(r, np.ones(20) / 20, mode="same")
        r = np.clip(r, 0, None)
        if r.sum() > 0:
            cands.append(r / r.sum())
    return cands


def _optimize(h0):
    global TAU
    h = np.asarray(h0, dtype=float).copy()
    s = h.sum()
    if s <= 0:
        return h0
    h = h / s
    TAU = 0.05
    f = _softmax_obj(h, TAU)
    for it in range(300):
        if time.time() > _DEADLINE:
            break
        g = _softmax_grad(h, TAU)
        h, f2 = _line_search(h, g, f)
        if f2 >= f - 1e-15:
            # cool down tau to sharpen
            TAU *= 0.9
            f = _softmax_obj(h, TAU)
            continue
        f = f2
        if TAU > 0.001 and it % 30 == 0:
            TAU *= 0.85
    return h


def construct():
    start = time.time()
    global TAU
    TAU = 0.05

    cands = _init_candidates()
    best_h, best_val = None, float("inf")
    for v in cands:
        vv = np.maximum(v, 0.0)
        if vv.sum() <= 0:
            continue
        sc = _score(vv)
        if sc < best_val:
            best_val, best_h = sc, vv.copy()

    if best_h is None:
        best_h, best_val = np.ones(N), _score(np.ones(N))

    # optimize each candidate, keep best
    for v in cands:
        if time.time() > _DEADLINE:
            break
        try:
            ho = _optimize(v)
            sc = _score(ho)
            if sc < best_val - 1e-12:
                best_val, best_h = sc, ho
        except Exception:
            continue

    # final local polish on best with fine tau
    if time.time() < _DEADLINE:
        TAU = 0.005
        try:
            hp = _optimize(best_h)
            sc = _score(hp)
            if sc < best_val - 1e-12:
                best_val, best_h = sc, hp
        except Exception:
            pass

    out = np.asarray(best_h, dtype=float)
    out = np.maximum(out, 0.0)
    s = out.sum()
    if s <= 0:
        return [1.0] * N
    out = out / s
    return [float(x) for x in out]
