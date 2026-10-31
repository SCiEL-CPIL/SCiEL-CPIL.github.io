"""Uncertainty inequality: tune Hermite-Gaussian coefficients to reduce the C4 bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, root

_T0 = time.time()
_DEADLINE = 210.0
_SAFETY = 8.0

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

# --- Hermite polynomials (physicists', H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}) ---
def _hermite(n, x):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    for k in range(1, n):
        H2 = 2.0 * x * H1 - 2.0 * k * H0
        H0, H1 = H1, H2
    return H1

# Polynomial P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12, with a12 chosen so P(0)=0.
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=1330560.
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 1330560.0

def _a12(a0, a4, a8):
    # P(0) = a0*1 + a4*12 + a8*1680 + a12*1330560 = 0
    return -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0

def _P(x, a0, a4, a8):
    a12 = _a12(a0, a4, a8)
    return (a0 * _hermite(0, x) + a4 * _hermite(4, x) +
            a8 * _hermite(8, x) + a12 * _hermite(12, x))

# Numerical uncertainty bound via a large grid + FFT (math convention e^{-2 pi i x xi}).
# Precompute grid once.
_N = 1 << 17  # 131072 points
_L = 40.0     # half-width; enough for e^{-x^2/2} tails
_X = np.linspace(-_L, _L, _N, endpoint=False)
_DX = _X[1] - _X[0]
_GAUSS = np.exp(-_X ** 2 / 2.0)

def _bound(coeffs):
    a0, a4, a8 = coeffs
    P = _P(_X, a0, a4, a8)
    g = P * _GAUSS
    g2 = g ** 2
    norm = np.sum(g2) * _DX
    if norm <= 0 or not np.isfinite(norm):
        return np.inf
    # A(f)^2
    A_f2 = np.sum(_X ** 2 * g2) * _DX / norm
    # Fourier transform via FFT (math convention)
    # FFT of g on uniform grid; dx=_DX, frequency spacing 1/(N dx)
    G = np.fft.fft(g) * _DX
    # shift so zero freq in middle
    G = np.fft.fftshift(G)
    # frequency axis
    freq = np.fft.fftfreq(_N, d=_DX)
    freq = np.fft.fftshift(freq)
    G2 = np.abs(G) ** 2
    normF = np.sum(G2) * (1.0 / (_N * _DX))  # Parseval-ish; use consistent spacing
    if normF <= 0 or not np.isfinite(normF):
        return np.inf
    A_F2 = np.sum(freq ** 2 * G2) * (1.0 / (_N * _DX)) / normF
    if A_f2 < 0 or A_F2 < 0:
        return np.inf
    return np.sqrt(max(A_f2, 0.0) * max(A_F2, 0.0))

def _bound_sq(coeffs):
    b = _bound(coeffs)
    return b * b if np.isfinite(b) else np.inf

# Multi-start global + local polish
def construct():
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _bound(best)
    if not np.isfinite(best_val):
        best_val = np.inf

    rng = np.random.default_rng(12345)

    # A set of diverse starting points
    starts = [best.copy()]
    # random starts in a reasonable box
    for _ in range(60):
        starts.append(rng.uniform(-5, 5, size=3))
    # a few scaled versions
    for s in (0.5, 2.0, -1.0):
        starts.append(s * best)

    for s in starts:
        if _time_left() < 3.0:
            break
        try:
            res = minimize(_bound_sq, s, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-14})
            if res.success or np.isfinite(res.fun):
                c = res.x
                v = _bound(c)
                if np.isfinite(v) and v < best_val:
                    best_val, best = v, c.copy()
        except Exception:
            pass

    # High-precision polish with BFGS/L-BFGS-B from the best point
    if _time_left() > 3.0:
        try:
            res = minimize(_bound_sq, best, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-12})
            if np.isfinite(res.fun):
                c = res.x
                v = _bound(c)
                if np.isfinite(v) and v < best_val:
                    best_val, best = v, c.copy()
        except Exception:
            pass

    # Refine with a tight Nelder-Mead from the polished point
    if _time_left() > 3.0:
        try:
            res = minimize(_bound_sq, best, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16})
            if np.isfinite(res.fun):
                c = res.x
                v = _bound(c)
                if np.isfinite(v) and v < best_val:
                    best_val, best = v, c.copy()
        except Exception:
            pass

    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
