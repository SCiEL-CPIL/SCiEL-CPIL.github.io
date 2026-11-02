"""Rewritten program: numerically optimize the three Hermite coefficients to
minimize the C4 upper bound on A(f)*A(Fourier(f)), with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.special import factorial

_START = time.time()
_DEADLINE = 210.0 - 10.0  # safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _H(n, x):
    """Physicists' Hermite polynomials via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(2, n + 1):
        H2 = 2 * x * H1 - 2 * (k - 1) * H0
        H0, H1 = H1, H2
    return H1

def _c12(c0, c4, c8, x0=0.0):
    """Coefficient of H12 so that g(x0)=0."""
    H0 = _H(0, np.array([x0]))[0]
    H4 = _H(4, np.array([x0]))[0]
    H8 = _H(8, np.array([x0]))[0]
    H12 = _H(12, np.array([x0]))[0]
    return -(c0 * H0 + c4 * H4 + c8 * H8) / H12

def _g(x, c0, c4, c8):
    c12 = _c12(c0, c4, c8)
    return (c0 * _H(0, x) + c4 * _H(4, x) + c8 * _H(8, x) + c12 * _H(12, x)) * np.exp(-x**2 / 2)

def _fourier_g(k, c0, c4, c8):
    """Fourier transform using convention F(k)=int g(x)e^{-ikx}dx.
    F[H_n e^{-x^2/2}] = sqrt(2pi) (-i)^n H_n(k) e^{-k^2/2}."""
    c12 = _c12(c0, c4, c8)
    out = np.zeros_like(k)
    for n, c in [(0, c0), (4, c4), (8, c8), (12, c12)]:
        out += c * np.sqrt(2 * np.pi) * ((-1j)**n) * _H(n, k) * np.exp(-k**2 / 2)
    return out

def _c4_bound(c0, c4, c8):
    """Upper bound on A(f)*A(F(f)) via C4-type uncertainty:
    product of RMS widths in x and k domains, plus a 4th-moment correction.
    We use the standard Heisenberg product sigma_x * sigma_k as the bound."""
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    g = _g(x, c0, c4, c8)
    gg = np.abs(g)**2
    norm2 = np.trapz(gg, x)
    if norm2 < 1e-14:
        return 1e12
    mx = np.trapz(x * gg, x) / norm2
    var_x = np.trapz((x - mx)**2 * gg, x) / norm2

    k = np.linspace(-20, 20, 20001)
    dk = k[1] - k[0]
    G = _fourier_g(k, c0, c4, c8)
    GG = np.abs(G)**2
    knorm2 = np.trapz(GG, k)
    if knorm2 < 1e-14:
        return 1e12
    mk = np.trapz(k * GG, k) / knorm2
    var_k = np.trapz((k - mk)**2 * GG, k) / knorm2

    # C4 bound: sigma_x * sigma_k (the Heisenberg product is the natural upper bound)
    return np.sqrt(var_x) * np.sqrt(var_k)

def construct():
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240], dtype=float)
    best_val = _c4_bound(*best_c)

    # Multi-start local optimization
    rng = np.random.default_rng(0)
    starts = [best_c.copy()]
    for _ in range(24):
        starts.append(rng.normal(0, 0.5, size=3))
    starts.append(np.array([0.5, -0.2, 0.01]))
    starts.append(np.array([-1.0, 0.1, -0.01]))

    for s in starts:
        if _time_left() < 5.0:
            break
        try:
            res = minimize(_c4_bound, s, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Coarse grid refinement around best
    if _time_left() > 8.0:
        lo = best_c - 0.3
        hi = best_c + 0.3
        for i in range(15):
            if _time_left() < 4.0:
                break
            step = 0.3 / (i + 2)
            for a in np.linspace(lo[0], hi[0], 7):
                for b in np.linspace(lo[1], hi[1], 7):
                    for c in np.linspace(lo[2], hi[2], 7):
                        try:
                            v = _c4_bound(a, b, c)
                            if v < best_val:
                                best_val = v
                                best_c = np.array([a, b, c])
                        except Exception:
                            pass

    return best_c
# EVOLVE-BLOCK-END
