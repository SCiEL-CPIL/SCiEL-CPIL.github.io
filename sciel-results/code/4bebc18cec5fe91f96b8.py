import numpy as np

N = 400
DEADLINE = 120.0
EPS = 1e-9


def _autoconv_bound(x):
    """Upper bound on C3 = max over lags of the signed autoconvolution of x."""
    x = np.asarray(x, dtype=float)
    n = x.size
    # full linear autocorrelation via FFT (zero-padded, no circular wrap)
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    X = np.fft.rfft(x, m)
    ac = np.fft.irfft(X * X.conj(), m)[:2 * n - 1]
    return float(ac.max())


def _grad_autoconv_bound(x):
    """Gradient of max_k sum_i x[i] x[i+k] w.r.t. x (subgradient of the max)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    X = np.fft.rfft(x, m)
    ac = np.fft.irfft(X * X.conj(), m)[:2 * n - 1]
    k = int(np.argmax(ac))
    # d/dx_j [sum_i x_i x_{i+k}] = x_{j-k} + x_{j+k}
    g = np.zeros(n)
    if k == 0:
        g[:] = 2.0 * x
    else:
        g[:n - k] += x[k:]
        g[k:] += x[:n - k]
    return g


def _raised_cosine_window(n, alpha=1.0):
    """Signed, band-limited window from a raised-cosine amplitude spectrum."""
    n = int(n)
    if n % 2:
        n += 1
    # DFT bin index (even-length DFT)
    k = np.arange(n)
    k = np.where(k <= n // 2, k, k - n)          # map to [-n/2, n/2)
    # raised-cosine amplitude over [-n/2, n/2)
    A = 0.5 * (1.0 + np.cos(2.0 * np.pi * k / n))
    A[n // 2] = 0.0                               # kill the Nyquist (DC) bin
    w = np.fft.irfft(A, n)                        # real, symmetric, zero-DC
    return w[:n]


def _polish(x0, max_iter=4000, tol=1e-11):
    """Projected gradient descent on the smooth quadratic autoconvolution bound."""
    import time
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_val = _autoconv_bound(x)
    f = best_val
    g = _grad_autoconv_bound(x)
    L = float(np.linalg.norm(g, 2)) + 1e-9        # Lipschitz estimate (2*sigma_max)
    lr = 2.0 / (L + 1e-12)
    it = 0
    t0 = time.time()
    while it < max_iter:
        if time.time() > DEADLINE - 5.0:
            break
        x = x - lr * g
        # soft projection toward zero mean (helps reduce the autocorrelation peak)
        x = x - x.mean()
        f = _autoconv_bound(x)
        if f < best_val - tol:
            best_val = f
            best_x = x.copy()
        g = _grad_autoconv_bound(x)
        it += 1
        if it % 250 == 0 and L > 0:
            L = max(L * 0.95, float(np.linalg.norm(g, 2)))
            lr = 2.0 / (L + 1e-12)
    return best_x, best_val


def construct():
    import time
    t0 = time.time()
    base = _raised_cosine_window(N, alpha=1.0)
    x, val = _polish(base)
    # safety: guarantee we return within budget
    if time.time() > DEADLINE - 2.0:
        return x
    return x
