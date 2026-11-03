import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 210.0
_SAFE = 20.0

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

# Precompute on a fixed grid
_N = 8192
_L = 14.0
_X = np.linspace(-_L, _L, _N)
_DX = _X[1] - _X[0]
_EXP = np.exp(-_X ** 2)
_H0 = _hermite(0, _X)
_H4 = _hermite(4, _X)
_H8 = _hermite(8, _X)
_H12 = _hermite(12, _X)
_X2 = _X ** 2

# Hermite values at 0 for the constraint
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _c12(c0, c4, c8):
    return -(c0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

# Precompute FFT frequency axis
_XI = 2.0 * np.pi * np.fft.fftfreq(_N, d=_DX)
_XI_S = np.fft.fftshift(_XI)
_XI2_S = _XI_S ** 2
_DXI = 2.0 * np.pi / (_N * _DX)

def _product(c0, c4, c8):
    c12 = _c12(c0, c4, c8)
    g = (c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12) * _EXP
    g2 = g * g
    norm = np.sum(g2) * _DX
    if norm <= 1e-300:
        return np.inf
    pos_var = np.sum(_X2 * g2) * _DX / norm
    # Fourier
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g)))
    G2 = G * G
    normF = np.sum(G2) * _DXI
    if normF <= 1e-300:
        return np.inf
    freq_var = np.sum(_XI2_S * G2) * _DXI / normF
    pos_rms = np.sqrt(pos_var) if pos_var > 0 else 0.0
    freq_rms = np.sqrt(freq_var) if freq_var > 0 else 0.0
    return pos_rms * freq_rms

def _objective(z):
    # z = [u, v] with c0 = 1, c4 = u, c8 = v
    u, v = z
    return _product(1.0, u, v)

def construct():
    global _T0, _DEADLINE, _SAFE
    deadline = _T0 + _DEADLINE - _SAFE
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    # seed in normalized form: c0=-1.13, so u=c4/c0, v=c8/c0
    s_u = seed[1] / seed[0]
    s_v = seed[2] / seed[0]
    best_z = np.array([s_u, s_v])
    best_val = _product(seed[0], seed[1], seed[2])
    if not np.isfinite(best_val):
        best_val = np.inf

    # Multi-start
    starts = [
        [s_u, s_v],
        [0.0, 0.0],
        [s_u * 1.5, s_v * 1.5],
        [s_u * 0.5, s_v * 0.5],
        [s_u + 0.01, s_v],
        [s_u - 0.01, s_v],
        [s_u, s_v + 0.001],
        [s_u, s_v - 0.001],
        [0.02, 0.0],
        [-0.02, 0.0],
        [s_u, 0.0005],
        [s_u, -0.0005],
    ]

    def time_ok():
        return time.time() < deadline

    for start in starts:
        if not time_ok():
            break
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.success or res.fun < best_val:
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_z = res.x.copy()
            # BFGS polish
            if time_ok():
                res2 = minimize(_objective, best_z, method='BFGS',
                                options={'maxiter': 2000, 'gtol': 1e-12})
                if np.isfinite(res2.fun) and res2.fun < best_val:
                    best_val = res2.fun
                    best_z = res2.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time_ok():
        try:
            res3 = minimize(_objective, best_z, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14})
            if np.isfinite(res3.fun) and res3.fun < best_val:
                best_val = res3.fun
                best_z = res3.x.copy()
        except Exception:
            pass

    u, v = best_z
    # Return in seed-like scale: c0 = -1.13 (match seed magnitude), or c0=1?
    # Bound is homogeneous, so scale is irrelevant to the evaluator's bound.
    # Return c0 = 1 for cleanliness, but ensure non-degenerate.
    c0 = 1.0
    c4 = u
    c8 = v
    return np.array([c0, c4, c8])
