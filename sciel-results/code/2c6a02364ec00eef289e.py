"""Erdos minimum overlap: spectral low-dimensional optimization.
Parameterize h via its DFT coefficients (conjugate symmetry, zero DC).
Optimize the EXACT max(C_k) objective over ~50 real spectral variables.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135


def _h_from_coeffs(coeffs):
    """Reconstruct h from spectral coefficients.
    coeffs layout: [re_1, im_1, re_2, im_2, ..., re_49, im_49, re_50]
    h_i = 0.5 + sum_{k=1}^{49} 2*re_k*cos(2*pi*k*i/N) + 2*im_k*sin(2*pi*k*i/N)
         + re_50 * cos(pi*i)
    """
    n = len(coeffs)
    # n should be 2*49 + 1 = 99
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, 50):
        re_k = coeffs[2 * (k - 1)]
        im_k = coeffs[2 * (k - 1) + 1]
        angle = 2.0 * np.pi * k * i / N
        h += 2.0 * (re_k * np.cos(angle) + im_k * np.sin(angle))
    # k=50 (Nyquist)
    re_50 = coeffs[-1]
    h += re_50 * np.cos(np.pi * i)  # cos(pi*i) = (-1)^i
    return h


def _coeffs_from_h(h):
    """Extract spectral coefficients from h (assuming sum = N/2)."""
    H = np.fft.rfft(h)
    coeffs = np.zeros(99)
    for k in range(1, 50):
        coeffs[2 * (k - 1)] = H[k].real
        coeffs[2 * (k - 1) + 1] = H[k].imag
    coeffs[-1] = H[50].real
    return coeffs


def _linear_overlap_values(h):
    """Compute C_k for k=0..N-1 using FFT-based autocorrelation."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _objective(coeffs):
    """Exact max(C_k) objective over spectral coefficients."""
    h = _h_from_coeffs(coeffs)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _gradient(coeffs, eps=1e-5):
    """Numerical gradient of the exact max objective."""
    grad = np.zeros_like(coeffs)
    base = _objective(coeffs)
    for j in range(len(coeffs)):
        pert = coeffs.copy()
        pert[j] += eps
        f_plus = _objective(pert)
        pert[j] -= 2 * eps
        f_minus = _objective(pert)
        grad[j] = (f_plus - f_minus) / (2 * eps)
    return grad


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        # Enforce sum constraint
        target = N / 2.0
        h = h - np.mean(h) + target / N
        h = np.clip(h, 0.0, 1.0)
        # Re-fix sum after clipping
        diff = target - np.sum(h)
        if abs(diff) > 1e-12:
            mask = (h > 0) & (h < 1)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) > 1e-15:
            idx = np.argmin(h) if diff > 0 else np.argmax(h)
            h[idx] += diff
            h = np.clip(h, 0.0, 1.0)

        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Generate diverse spectral initializations
    i = np.arange(N)
    starts = []

    # 1. Cosine wave freq 1
    h1 = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_coeffs_from_h(h1 - np.mean(h1) + N / 2.0 / N))

    # 2. Cosine wave freq 2
    h2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    starts.append(_coeffs_from_h(h2 - np.mean(h2) + N / 2.0 / N))

    # 3. Sine wave freq 1
    h3 = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_coeffs_from_h(h3 - np.mean(h3) + N / 2.0 / N))

    # 4. Triangular wave
    h4 = 0.5 + 0.5 * (1.0 - 2.0 * np.abs((2.0 * i / N) - 1.0))
    starts.append(_coeffs_from_h(h4 - np.mean(h4) + N / 2.0 / N))

    # 5. Sawtooth
    h5 = 0.5 + 0.5 * (2.0 * (i / N) - 1.0)
    starts.append(_coeffs_from_h(h5 - np.mean(h5) + N / 2.0 / N))

    # 6-10: Random spectral initializations
    for _ in range(5):
        c = np.random.normal(0, 0.1, 99)
        # Scale to keep h in [0,1] roughly
        h_test = _h_from_coeffs(c)
        if h_test.max() > 1.0 or h_test.min() < 0.0:
            scale = 0.9 / max(abs(h_test.max() - 0.5), abs(h_test.min() - 0.5), 1e-10)
            c *= scale
        starts.append(c)

    # 11-15: Random time-domain converted to spectral
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = h_rand - np.mean(h_rand) + N / 2.0 / N
        starts.append(_coeffs_from_h(h_rand))

    # Evaluate all starts
    for c in starts:
        h = _h_from_coeffs(c)
        _update_best(h)

    # Optimize each start with SLSQP on spectral coefficients
    for c0 in starts:
        if time.time() - t0 > _DEADLINE - 15:
            break
        try:
            res = minimize(
                _objective,
                c0.copy(),
                jac=_gradient,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.fun < best_score or (res.success and _h_from_coeffs(res.x).min() >= -1e-6):
                h_res = _h_from_coeffs(res.x)
                _update_best(h_res)
        except Exception:
            pass

    # Final polish on best with tighter tolerance
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        c_best = _coeffs_from_h(best_h)
        try:
            res = minimize(
                _objective,
                c_best.copy(),
                jac=_gradient,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                h_res = _h_from_coeffs(res.x)
                _update_best(h_res)
        except Exception:
            pass

    if best_h is None:
        best_h = np.full(N, 0.5)

    # Final sum fix
    target = N / 2.0
    diff = target - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 0) & (best_h < 1)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
    diff = target - np.sum(best_h)
    if abs(diff) > 1e-15:
        idx = np.argmin(best_h) if diff > 0 else np.argmax(best_h)
        best_h[idx] += diff
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
