"""Optimize signed step heights to minimize max |autoconvolution| / (sum)^2."""
import time
import numpy as np

N = 400

_DEADLINE = 150.0  # wall-clock kill limit
_SAFETY = 20.0     # return this many seconds before the limit


def _autoconv(x):
    """Full autoconvolution C[k] = sum_i x[i]*x[i-k] for k in [-(N-1), N-1]."""
    # Use FFT for speed: autoconv = ifft(fft(x)^2)
    n = len(x)
    m = 1
    while m < 2 * n - 1:
        m *= 2
    X = np.fft.rfft(x, m)
    c = np.fft.irfft(X * X, m)
    # c[0] = C(0), c[1..] = C(1..), c[m-1..] = C(-1..)
    return c


def _objective(x):
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return np.max(np.abs(c)) / (s * s)


def _scale_to_sum1(x):
    s = x.sum()
    if abs(s) < 1e-12:
        return x
    return x / s


def _initial_rudin_shapiro(n):
    """Rudin-Shapiro-like balanced +/- sequence."""
    x = np.zeros(n)
    a = np.array([1.0])
    b = np.array([1.0])
    # Build RS sequence: a_{k+1} = a_k + b_k, b_{k+1} = a_k - b_k (reversed b)
    # Simpler: generate via recursion producing +/-1
    def rs(k):
        if k == 0:
            return np.array([1.0])
        prev = rs(k - 1)
        return np.concatenate([prev, prev * np.concatenate([np.ones(len(prev)//2), -np.ones(len(prev)-len(prev)//2)])])
    # Use iterative RS: start [1], each step: new = concat(old, old * sign_pattern)
    seq = np.array([1.0])
    while len(seq) < n:
        # Rudin-Shapiro step: a <- a, b <- a - reverse(b) ... use standard:
        # a_{k+1} = a_k + b_k ; b_{k+1} = a_k - b_k (elementwise, b reversed)
        a = seq
        b = np.concatenate([a[:len(a)//2], -a[len(a)//2:]])
        seq = np.concatenate([a, a - b])
    return seq[:n]


def _initial_barkerish(n):
    """Random +/-1 with some structure."""
    rng = np.random.default_rng(12345)
    return rng.choice([-1.0, 1.0], size=n)


def _subgradient_step(x, c, lr, s):
    """One subgradient step on max_k |c_k|/s^2 w.r.t. x.
    c = autoconv(x). dc/dx is a circulant-ish matrix. We approximate via
    the active lag(s) where |c| is maximal."""
    n = len(x)
    # Find active lags
    idx = np.argmax(np.abs(c))
    # c[idx] corresponds to lag: if idx < n: lag = idx; else lag = idx - (2n-2)
    m = len(c)
    if idx < n:
        lag = idx
    else:
        lag = idx - (m - 1)
    sign = np.sign(c[idx])
    # dC_lag/dx: for lag L >= 0, C_L = sum_i x[i+L] x[i]; dC_L/dx[j] = x[j-L] (if j>=L) + x[j+L] (if j+L<n)
    grad = np.zeros(n)
    if lag >= 0:
        grad[lag:] += x[:n - lag]
        grad[:n - lag] += x[lag:]
    else:
        L = -lag
        grad[:n - L] += x[L:]
        grad[L:] += x[:n - L]
    grad *= sign
    # Gradient of 1/s^2 factor: objective = max|c|/s^2. d(obj)/dx = (grad*sign_active)/s^2 - 2*max|c|*x/s^3
    obj = np.max(np.abs(c))
    grad_obj = grad / (s * s) - 2.0 * obj * x / (s ** 3)
    return x - lr * grad_obj


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _SAFETY

    # Start from several diverse initializations
    candidates = []
    for init in [_initial_rudin_shapiro(N), _initial_barkerish(N)]:
        x = _scale_to_sum1(init.astype(float)).copy()
        candidates.append(x)

    best_x = candidates[0].copy()
    best_val = _objective(best_x)

    # Subgradient descent with adaptive step size
    x = best_x.copy()
    lr = 0.05
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        s = x.sum()
        c = _autoconv(x)
        x = _subgradient_step(x, c, lr, s)
        # Renormalize to keep sum ~ 1 (helps stability)
        x = _scale_to_sum1(x)
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            lr = max(lr * 1.05, 1e-4)
        else:
            lr = max(lr * 0.9, 1e-6)
        # Occasional random restart if stuck
        if n_iter % 400 == 0:
            rng = np.random.default_rng(n_iter)
            x = _scale_to_sum1(rng.choice([-1.0, 1.0], size=N)).copy()

    # Final polish: a few coordinate-descent-ish local tweaks on best_x
    x = best_x.copy()
    for _ in range(200):
        if time.time() > deadline:
            break
        s = x.sum()
        c = _autoconv(x)
        idx = np.argmax(np.abs(c))
        m = len(c)
        lag = idx if idx < N else idx - (m - 1)
        sign = np.sign(c[idx])
        # Try flipping the sign of the element most correlated with reducing |c_lag|
        # Approximate: adjust x at positions contributing most to c_lag
        if lag >= 0:
            contrib = x[:N - lag] * x[lag:]
        else:
            L = -lag
            contrib = x[L:] * x[:N - L]
        # pick index with largest |contrib|
        j = np.argmax(np.abs(contrib))
        # flip sign of that element
        x2 = x.copy()
        x2[j] = -x2[j]
        x2 = _scale_to_sum1(x2)
        v2 = _objective(x2)
        if v2 < best_val:
            best_val = v2
            best_x = x2.copy()
            x = x2

    return best_x


# EVOLVE-BLOCK-START
# (entry point construct defined above; N fixed)
# EVOLVE-BLOCK-END
