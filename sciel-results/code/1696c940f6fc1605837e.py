"""Direct construction of step heights to minimize the third autocorrelation inequality value.

The task value is abs(2*N*max(conv(h,h))/sum(h)^2). The maximum of the full autoconvolution
of a finite sequence is always at index N-1 (zero lag) and equals sum(h^2) >= 0. Therefore the
objective reduces to minimizing sum(h^2)/sum(h)^2, which is minimized when all entries are
equal (value -> 2.0). We construct explicit low-sidelobe structures (Barker-like, Zadoff-Chu,
Golay complementary pairs) and verify via FFT convolution, then polish with coordinate descent
on the smooth log-sum-exp surrogate of the max-conv value."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 135.0

_i = np.arange(N, dtype=float)


def _value(h):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))


def _fft_convolve(h):
    size = 2 * N
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)[:2 * N - 1]


def _smooth_obj(h, beta=80.0):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 0)
    lse = m + np.log(np.sum(np.exp(scaled)) + 1e-300) / beta
    return abs(2.0 * N * lse / (s * s))


def _barker_h(rate=2.0, amp=1.0, base=0.0):
    # A chirp whose autocorrelation has a sharp main lobe and low sidelobes.
    ang = 2.0 * np.pi * rate * _i * (_i + 1) / N
    return base + amp * np.cos(ang)


def _windowed_zadoff_chu_h(rate, amp=1.0, base=0.0, window='hanning'):
    # Windowed Zadoff-Chu to reduce spectral leakage / edge effects
    ang = np.pi * rate * _i * (_i + 1) / N
    zc = amp * np.cos(ang)
    if window == 'hanning':
        w = np.hanning(N)
    elif window == 'blackman':
        w = np.blackman(N)
    else:
        w = np.ones(N)
    return base + zc * w


def _golay_pair_h(p1, r1, a1, p2, r2, a2, base=0.0):
    ang1 = 2.0 * np.pi * r1 * _i * (_i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * _i * (_i + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)


def _random_signs(rng):
    return rng.choice([-1.0, 1.0], size=N)


def _coord_descent(h0, max_iter=80, step=0.02, beta=80.0, deadline_fn=None):
    h = h0.copy()
    best_v = _smooth_obj(h, beta)
    for _ in range(max_iter):
        if deadline_fn is not None and not deadline_fn():
            break
        improved = False
        for i in range(N):
            if deadline_fn is not None and not deadline_fn():
                break
            for delta in (step, -step):
                ht = h.copy()
                ht[i] += delta
                vt = _smooth_obj(ht, beta)
                if vt < best_v:
                    h = ht
                    best_v = vt
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return h, best_v


def _dct_refine(h0, beta=100.0, maxiter=200, deadline_fn=None):
    # Parameterize h = base + sum_k c_k * cos(pi*(2n+1)*k/(2N)) using first few DCT modes.
    K = 32
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms

    # Initial coefficients via least squares against h0 - mean
    mean0 = float(np.mean(h0))
    target = h0 - mean0
    try:
        coeffs0 = np.linalg.lstsq(basis, target, rcond=None)[0]
    except Exception:
        coeffs0 = np.zeros(K)

    def obj(x):
        if deadline_fn is not None and not deadline_fn():
            return np.inf
        base = x[-1]
        c = x[:-1]
        h = base + basis @ c
        return _smooth_obj(h, beta)

    x0 = np.concatenate([coeffs0, [mean0]])
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.x is not None:
            base = res.x[-1]
            c = res.x[:-1]
            return base + basis @ c
    except Exception:
        pass
    return h0


def construct():
    global _t0
    _t0 = time.time()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    best_h = np.ones(N)
    best_v = _value(best_h)

    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    rng = np.random.default_rng(123)

    # Direct structural candidates
    candidates = []

    # 1. Windowed Zadoff-Chu with several rates and windows
    for rate in [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
        for win in ['hanning', 'blackman']:
            candidates.append(_windowed_zadoff_chu_h(rate, window=win))

    # 2. Windowed Chirp (quadratic phase) with various rates
    for rate in [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        ang = 2.0 * np.pi * rate * _i * (_i + 1) / N
        chirp = np.cos(ang)
        for win in ['hanning', 'blackman']:
            if win == 'hanning':
                w = np.hanning(N)
            else:
                w = np.blackman(N)
            candidates.append(chirp * w)

    # 3. Golay complementary pairs (two chirps with different rates)
    for r1 in [2, 4, 6, 8, 10, 12]:
        for r2 in [3, 5, 7, 9, 11, 13]:
            if abs(r1 - r2) < 2:
                continue
            candidates.append(_golay_pair_h(0.0, r1, 1.0, np.pi / 2, r2, 1.0))

    # 4. Random signs (baseline diversity)
    for _ in range(20):
        candidates.append(_random_signs(rng))

    # Evaluate all candidates
    for h in candidates:
        if not deadline_ok():
            break
        update_best(h)

    # DCT spectral refinement on the best candidate
    if deadline_ok():
        h_ref = _dct_refine(best_h, beta=100.0, maxiter=300, deadline_fn=deadline_ok)
        update_best(h_ref)

    # Full-dimensional L-BFGS-B polish on the smooth surrogate
    if deadline_ok():
        def full_obj(h):
            if deadline_fn is not None and not deadline_fn():
                return np.inf
            return _smooth_obj(h, beta=100.0)
        
        try:
            res = minimize(full_obj, best_h, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    # Coordinate descent polish on full-dimensional h
    if deadline_ok():
        h_cd, _ = _coord_descent(best_h, max_iter=100, step=0.01, beta=100.0, deadline_fn=deadline_ok)
        update_best(h_cd)

    # Second round: refine the current best with a different DCT seed
    if deadline_ok():
        h2 = best_h + rng.normal(0, 0.05, N)
        h2_ref = _dct_refine(h2, beta=120.0, maxiter=200, deadline_fn=deadline_ok)
        update_best(h2_ref)

    # Final full-dimensional polish
    if deadline_ok():
        def full_obj2(h):
            if deadline_fn is not None and not deadline_fn():
                return np.inf
            return _smooth_obj(h, beta=150.0)
            
        try:
            res = minimize(full_obj2, best_h, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    # Safety: ensure valid output
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
