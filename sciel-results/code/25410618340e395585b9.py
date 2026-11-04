"""
Search for nonnegative step heights maximizing (L2 norm)^2 / (L1 norm * sup autoconvolution).

Strategy:
- The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
- For K non-interacting spikes (Sidon set), R = K.
- We use a greedy Sidon set construction to find a sparse initial pattern,
  then refine with L-BFGS-B using a smooth-max objective and complex parameterization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_dx = (0.25 - (-0.25)) / N  # 0.5/50 = 0.01
_DEADLINE = 115.0  # safety margin before 120s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _elapsed()


def _autoconv_peak(h):
    """Compute supremum norm of autoconvolution using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return np.max(conv)


def _ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0)
    l2sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 < 1e-15 or peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _smooth_max(x, sigma=0.1):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 1e-30:
        return 1e-30
    x_shifted = x - x_max
    return x_max + sigma * np.log(np.sum(np.exp(x_shifted / sigma))) / len(x)


def _objective(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0] ** 2 + g[:, 1] ** 2

    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1

    l2sq = np.sum(h_norm * h_norm)

    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h_norm, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    peak = _smooth_max(conv, sigma=0.05)

    if peak < 1e-30:
        return 0.0

    return -l2sq / peak


def _optimize_from_g(g0, max_iters=300, maxfun=1500):
    """Run L-BFGS-B optimization on complex parameters."""
    result = minimize(
        _objective,
        g0,
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxfun}
    )
    g_opt = result.x.reshape(N, 2)
    h_opt = g_opt[:, 0] ** 2 + g_opt[:, 1] ** 2
    h_opt = np.maximum(h_opt, 0)
    return h_opt


def _h_to_g(h):
    """Convert h to g (real part = sqrt(h), imag = 0)."""
    h = np.maximum(h, 0)
    g_real = np.sqrt(h)
    g_imag = np.zeros(N)
    return np.concatenate([g_real, g_imag])


def _sidon_set(n, max_size):
    """Generate a Sidon set (B2 set) in {0,...,n-1} of given size."""
    # Greedy construction
    S = []
    for i in range(n):
        if len(S) >= max_size:
            break
        # Check if i can be added
        valid = True
        for j in range(len(S)):
            for k in range(j + 1, len(S)):
                if S[j] + S[k] == i + S[j] or S[j] + S[k] == i + S[k]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(i)
    return S


def _best_sidon_pattern():
    """Find the best Sidon set pattern by evaluating ratio."""
    best_h = None
    best_val = -np.inf

    # Try different sizes of Sidon sets
    for size in range(5, 12):
        S = _sidon_set(N, size)
        if len(S) < 4:
            continue
        h = np.zeros(N)
        for idx in S:
            h[idx] = 1.0
        l1 = np.sum(h)
        if l1 > 0:
            h = h / l1
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    return best_h, best_val


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_val = -np.inf

    def consider(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
        return v

    # Initial candidates
    candidates = []

    # Uniform
    candidates.append(np.ones(N))

    # Single spike at center
    for pos in [0, N // 2, N - 1]:
        s = np.zeros(N)
        s[pos] = 1.0
        candidates.append(s)

    # Narrow blocks
    for w in [1, 2, 3, 4, 5]:
        for pos in [0, N // 2 - w // 2, N - w]:
            if pos >= 0 and pos + w <= N:
                b = np.zeros(N)
                b[pos:pos + w] = 1.0
                candidates.append(b)

    # Sidon set pattern
    sidon_h, sidon_val = _best_sidon_pattern()
    if sidon_h is not None:
        candidates.append(sidon_h)

    # Evaluate all candidates
    for c in candidates:
        consider(c)

    # Local optimization from each promising start using L-BFGS-B
    elapsed = time.time() - _start
    per_start = max(0.5, (_DEADLINE - 5.0 - elapsed) / max(1, len(candidates)))

    for c in candidates:
        if time.time() > _DEADLINE - 5.0:
            break

        # Try different scales
        for scale in [1.0, 0.1, 0.01]:
            if time.time() > _DEADLINE - 5.0:
                break
            x0 = c * scale
            g0 = _h_to_g(x0)
            try:
                h_opt = _optimize_from_g(g0)
                consider(h_opt)
            except Exception:
                pass

    # A final coordinate-descent refinement on the best found, if time remains.
    if best_h is not None and time.time() < _DEADLINE - 5.0:
        h = best_h.copy()
        step = 1e-3
        for it in range(2000):
            if time.time() > _DEADLINE - 5.0:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for delta in (step, -step, 2 * step, 0.5 * step):
                    cand = h.copy()
                    cand[i] = max(0.0, old + delta)
                    if _ratio(cand) > best_val:
                        h = cand
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-7:
                    break
        consider(h)

    if best_h is None:
        return [1.0] * N

    # Return as a plain list of floats
    return [float(x) for x in best_h]
