"""Fresh design: signed step function on 400 equal intervals minimizing the
autoconvolution peak-sidelobe (the C3 upper bound)."""
import time
import numpy as np

N = 400

# ---------- metric: peak sidelobe of discrete autoconvolution ----------
def _peak_sidelobe(a):
    c = np.convolve(a, a, mode='full')
    peak = c[N - 1]
    if peak <= 0:
        return 1e18
    # exclude the peak bin
    tail = np.concatenate((c[:N - 1], c[N:]))
    return float(np.max(tail) / peak)


def _kaiser_like(beta, taper):
    n = np.arange(N)
    t = (n - (N - 1) / 2.0) / ((N - 1) / 2.0)          # in [-1,1]
    x = np.sqrt(np.maximum(1.0 - t * t, 0.0))
    k = np.i0(beta * x) / np.i0(beta)
    g = np.exp(-t * t / (2.0 * taper * taper))
    a = k * g
    s = a.sum()
    if s <= 0:
        a = np.ones(N) / N
    else:
        a = a / s
    return a


def _refine(a0, budget):
    """Projected-gradient descent on peak-sidelobe; keeps a>=0, sum=1."""
    a = a0.astype(np.float64).copy()
    a = np.clip(a, 0.0, None)
    a /= a.sum()
    best = _peak_sidelobe(a)
    a_best = a.copy()
    lr = 1e-2
    for _ in range(20000):
        if time.time() > budget:
            break
        c = np.convolve(a, a, mode='full')
        peak = c[N - 1]
        tail = np.concatenate((c[:N - 1], c[N:]))
        m = np.max(tail)
        if m / peak < best:
            best = m / peak
            a_best = a.copy()
        # gradient of (max_tail)/peak wrt a
        idx = np.argmax(tail)
        g = np.zeros(2 * N - 1)
        g[idx] = 1.0
        grad_c = 2.0 * np.convolve(a, g, mode='full')
        grad = grad_c / peak - (m / peak) * (grad_c[N - 1] / peak)
        a = a - lr * grad
        a = np.clip(a, 0.0, None)
        s = a.sum()
        if s <= 0:
            a = a_best.copy()
            break
        a = a / s
        # adaptive learning rate: shrink if it oscillates
        if not np.isfinite(best):
            break
    return a_best, best


def construct():
    start = time.time()
    deadline = start + 110.0          # safety margin under 120s

    # ---- coarse grid search over beta and taper ----
    best = np.inf
    a_best = None
    betas = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 26.0, 34.0])
    tapers = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.50, 2.50])
    for beta in betas:
        if time.time() > deadline:
            break
        for taper in tapers:
            a = _kaiser_like(beta, taper)
            v = _peak_sidelobe(a)
            if v < best:
                best = v
                a_best = a.copy()

    # ---- refine best candidate with projected gradient ----
    if a_best is not None and time.time() < deadline - 5.0:
        a_ref, v_ref = _refine(a_best, deadline - 5.0)
        if v_ref < best:
            best = v_ref
            a_best = a_ref

    if a_best is None:
        a_best = np.ones(N) / N
        best = _peak_sidelobe(a_best)

    # explicit deadline guard: return best found
    return a_best
