"""Golay complementary pair + DCT spectral shaping + smooth-max refinement.

Key insight: Golay complementary pairs (a, b) satisfy
    autocorr(a) + autocorr(b) = 2*delta_0
so the combined signal h = a + b has a much flatter autoconvolution peak.
We generate Golay pairs, then refine with DCT spectral shaping.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 400
_t0 = time.time()
_DEADLINE = _t0 + 130.0  # safety margin inside 150 s kill


def _value(h):
    """Raw task value (lower is better)."""
    s = float(np.sum(h))
    if s == 0.0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _fft_convolve(h):
    """Full autoconvolution via FFT (fast)."""
    n = 2 * len(h) - 1
    f = np.fft.rfft(h, n=n)
    return np.fft.irfft(f * f, n=n)


def _smooth_max_value(h, beta=200.0):
    """Smooth approximation of the peak autoconvolution objective."""
    s = float(np.sum(h))
    if s == 0.0 or not np.isfinite(s):
        return np.inf
    conv = _fft_convolve(h)
    # Softmax over conv entries to approximate max
    max_c = np.max(conv)
    if max_c == 0:
        return 0.0
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    smooth_max = max_c + np.log(np.sum(weights)) / beta
    return abs(2.0 * N * smooth_max / (s * s))


def _dct_basis_matrix(N, K):
    """DCT-II basis matrix (N x K) for the first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    B = np.cos(np.pi * (2 * n[:, None] + 1) * k[None, :] / (2.0 * N))
    B[0, :] *= np.sqrt(1.0 / N)
    B[1:, :] *= np.sqrt(2.0 / N)
    return B


def _dct_h(coeffs):
    """Reconstruct h from DCT coefficients (first K modes)."""
    K = len(coeffs)
    B = _dct_basis_matrix(N, K)
    return B @ coeffs


def _dct_obj(x, K):
    """Objective in DCT coefficient space."""
    h = _dct_h(x)
    return _value(h)


def _dct_smooth_obj(x, K, beta=200.0):
    """Smooth objective in DCT coefficient space."""
    h = _dct_h(x)
    return _smooth_max_value(h, beta)


def _golay_complementary_h(rng, length):
    """Generate a Golay complementary pair and return h = a + b."""
    # Build Golay sequences via recursive construction
    def golay_seq(n, rng):
        """Generate a Golay sequence of length 2^n."""
        if n == 0:
            return np.array([1.0])
        a = golay_seq(n - 1, rng)
        # Random sign flip for diversity
        sign = 1.0 if rng.random() < 0.5 else -1.0
        b = a * sign
        # Golay recursion: (a, b) -> (a+b, a-b) or (a+b, -(a-b))
        c = a + b
        d = a - b
        # Randomly choose between the two valid recursions
        if rng.random() < 0.5:
            return np.concatenate([c, d])
        else:
            return np.concatenate([c, -d])

    # Get length 256 Golay pair, then pad/truncate to 400
    base_len = 256  # 2^8
    a = golay_seq(8, rng)
    b = golay_seq(8, rng)

    # Scale to length N by zero-padding and adding a windowed chirp for the rest
    h = np.zeros(N)
    h[:base_len] = a
    h[base_len:2 * base_len] = b

    # Add a windowed Zadoff-Chu-like sequence for the remaining part
    if N > 2 * base_len:
        rem = N - 2 * base_len
        zc = np.exp(1j * np.pi * np.arange(rem) * (rem + 1) / (2.0 * rem))
        window = np.hanning(rem)
        h[2 * base_len:] = np.real(zc) * window

    # Normalize sum to 1 for scale invariance
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s * N
    return h


def _golay_obj(x, rng):
    """Objective for Golay-based parameterization."""
    h = _golay_complementary_h(x, N)
    return _value(h)


def _golay_smooth_obj(x, rng, beta=200.0):
    """Smooth objective for Golay-based parameterization."""
    h = _golay_complementary_h(x, N)
    return _smooth_max_value(h, beta)


def _windowed_zadoff_chu_h(rng, N):
    """Windowed Zadoff-Chu sequence."""
    M = N
    n = np.arange(M)
    # Zadoff-Chu: x[n] = exp(-j*pi*n*(n+1)/M)
    zc = np.exp(-1j * np.pi * n * (n + 1) / M)
    # Apply a raised-cosine window to reduce spectral leakage
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (M - 1))) if M > 1 else np.ones(M)
    h = np.real(zc) * window
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s * N
    return h


def _zadoff_chu_obj(x):
    """Objective for Zadoff-Chu parameterization."""
    h = _windowed_zadoff_chu_h(x, N)
    return _value(h)


def _refine_dct(h, iters, K=40, beta=200.0):
    """Refine h using DCT coefficient optimization with smooth max."""
    # Project to DCT space
    B = _dct_basis_matrix(N, K)
    # Least-squares projection
    coeffs = np.linalg.lstsq(B, h, rcond=None)[0]

    def f(x):
        return _dct_smooth_obj(x, K, beta)

    res = minimize(f, coeffs, method='L-BFGS-B',
                   options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8})
    h_new = _dct_h(res.x)
    return h_new


def construct():
    best_h = np.ones(N)
    best_val = 2.0
    rng = np.random.default_rng(42)

    # --- Phase 1: Golay complementary pairs (primary strategy) ---
    n_golay = 8
    for i in range(n_golay):
        if time.time() > _DEADLINE - 50:
            break
        h = _golay_complementary_h(rng, N)
        v = _value(h)
        if np.isfinite(v) and v < best_val:
            best_val, best_h = v, h.copy()

    # --- Phase 2: Windowed Zadoff-Chu sequences ---
    if time.time() < _DEADLINE - 40:
        for i in range(6):
            if time.time() > _DEADLINE - 35:
                break
            h = _windowed_zadoff_chu_h(rng, N)
            # Add random phase rotation
            phase = rng.uniform(0, 2 * np.pi)
            n = np.arange(N)
            rot = np.cos(phase + 2 * np.pi * rng.uniform(0, 1) * n / N)
            h = h * rot
            s = np.sum(h)
            if abs(s) > 1e-12:
                h = h / s * N
            v = _value(h)
            if np.isfinite(v) and v < best_val:
                best_val, best_h = v, h.copy()

    # --- Phase 3: DCT spectral refinement of best candidate ---
    if time.time() < _DEADLINE - 25:
        for K in [20, 40, 60]:
            if time.time() > _DEADLINE - 20:
                break
            try:
                h_refined = _refine_dct(best_h, iters=200, K=K, beta=200.0)
                v = _value(h_refined)
                if np.isfinite(v) and v < best_val:
                    best_val, best_h = v, h_refined.copy()
            except Exception:
                continue

    # --- Phase 4: Subgradient-style local refinement ---
    if time.time() < _DEADLINE - 10:
        h = best_h.copy()
        s = np.sum(h)
        if abs(s) > 1e-12:
            h = h / s * N
        best_local = _value(h)
        lr = 1e-4
        for it in range(3000):
            if time.time() > _DEADLINE - 8:
                break
            conv = _fft_convolve(h)
            max_idx = np.argmax(conv)
            # Gradient of max(conv) w.r.t. h: h shifted by max_idx
            if max_idx >= N:
                shift = max_idx - N + 1
                g = np.zeros(N)
                g[:N - shift] = h[shift:]
            else:
                g = np.zeros(N)
                g[max_idx:] = h[:N - max_idx]
            # Also subtract mean to maintain sum constraint
            g = g - np.mean(g)
            h_new = h - lr * g
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                lr *= 0.5
                continue
            h_new = h_new / s_new * N
            v = _value(h_new)
            if np.isfinite(v) and v < best_local:
                best_local = v
                h = h_new
                lr *= 1.02
            else:
                lr *= 0.8

        if best_local < best_val:
            best_val, best_h = best_local, h

    # --- Final safety: return seed if nothing improved ---
    final = _value(best_h)
    if not np.isfinite(final) or final >= 2.0:
        return np.ones(N)
    return best_h
