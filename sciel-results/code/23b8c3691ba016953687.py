import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

_START = time.time()
_DEADLINE = _START + 165.0  # safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# Hermite polynomial coefficients (physicists'), indexed by power of x
_H0 = np.array([1.0])  # H0 = 1
_H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])  # H4 = 16x^4 - 48x^2 + 12
_H8 = np.array([112.0, 0.0, -1344.0, 0.0, 2688.0, 0.0, -1280.0, 0.0, 256.0])
_H12 = np.array([9216.0, 0.0, -96768.0, 0.0, 253440.0, 0.0, -282240.0, 0.0, 153600.0, 0.0, -39936.0, 0.0, 4096.0])

def _b_coeffs(c0, c4, c8):
    """Return b[0..6] where b[j] is coeff of x^{2j} in the polynomial part."""
    c12 = -(c0 + 12.0*c4 + 112.0*c8) / 9216.0
    b = np.zeros(7)
    # x^0: c0*1 + c4*12 + c8*112 + c12*9216 = 0 (by construction)
    b[0] = 0.0
    # x^2
    b[1] = c4*(-48.0) + c8*(-1344.0) + c12*(-96768.0)
    # x^4
    b[2] = c4*16.0 + c8*2688.0 + c12*253440.0
    # x^6
    b[3] = c8*(-1280.0) + c12*(-282240.0)
    # x^8
    b[4] = c8*256.0 + c12*153600.0
    # x^10
    b[5] = c12*(-39936.0)
    # x^12
    b[6] = c12*4096.0
    return b

# Precompute gamma values for the integral
_n_max = 25  # max degree in u = x^2 for the 4th power
_gamma_vals = np.array([gamma(n + 0.5) / (2.0 ** (n + 0.5)) for n in range(_n_max + 1)])

def _poly4_integral(coeffs_u):
    """Compute ∫ (Σ coeffs_u[m] u^m)^4 e^{-2x^2} dx where u = x^2."""
    # Convolve 4 times
    d1 = np.convolve(coeffs_u, coeffs_u)
    d2 = np.convolve(d1, d1)
    # d2[n] is coeff of u^n in the 4th power
    # Integral = Σ d2[n] * ∫ u^n e^{-2x^2} dx = Σ d2[n] * Γ(n+1/2) / 2^{n+1/2}
    # But wait: ∫_{-∞}^{∞} x^{2n} e^{-2x^2} dx = Γ(n+1/2) / 2^{n+1/2} * ... let me verify
    # Actually: ∫ x^{2n} e^{-ax^2} dx = Γ(n+1/2) / (a^{n+1/2} * 2) ... no
    # ∫_{-∞}^{∞} x^{2n} e^{-ax^2} dx = (2n-1)!! * sqrt(π) / (2^n * a^{n+1/2})
    # = Γ(n+1/2) * 2 / (a^{n+1/2} * sqrt(π)) * ... let me just use the formula directly
    # With a=2: ∫ x^{2n} e^{-2x^2} dx = Γ(n+1/2) / (2^{n+1/2} * sqrt(2/π)) ... 
    # Actually the seed code uses gamma(n+0.5)/2^(n+0.5), so I'll match that
    n = np.arange(len(d2))
    if len(d2) > len(_gamma_vals):
        ext = np.array([gamma(n[i] + 0.5) / (2.0 ** (n[i] + 0.5)) for i in range(len(_gamma_vals), len(d2))])
        vals = np.concatenate([_gamma_vals, ext])
    else:
        vals = _gamma_vals[:len(d2)]
    return float(np.dot(d2, vals))

# Precompute P_k polynomials
def _build_P():
    P = [[1.0]]
    for k in range(12):
        Pk = P[k]
        dpk = [i*Pk[i] for i in range(1, len(Pk))]
        xPk = [0.0] + Pk
        L = max(len(dpk), len(xPk))
        dpk = dpk + [0.0]*(L-len(dpk))
        xPk = xPk + [0.0]*(L-len(xPk))
        Pk1 = [dpk[i] - xPk[i] for i in range(L)]
        P.append(Pk1)
    return P

_P = _build_P()

def _R_coeffs_u(b):
    R = np.zeros(13)
    for j in range(7):
        sign = (-1.0)**j
        Pk = np.array(_P[2*j])
        R[:len(Pk)] += b[j]*sign*Pk
    R_u = R[0::2]
    return R_u

def _raw_value(c0, c4, c8):
    b = _b_coeffs(c0, c4, c8)
    A_g = _poly4_integral(b)
    R_u = _R_coeffs_u(b)
    A_gf = _poly4_integral(R_u)
    return A_g * A_gf

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
_RAW_SEED = _raw_value(*_SEED)

def _value(c0, c4, c8):
    return _raw_value(c0, c4, c8) / _RAW_SEED

def _objective(x):
    return _value(x[0], x[1], x[2])

def _gradient(x):
    """Numerical gradient using central differences."""
    eps = 1e-8
    g = np.zeros(3)
    f0 = _objective(x)
    for i in range(3):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (_objective(xp) - _objective(xm)) / (2.0 * eps)
    return g

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _value(*_SEED)
    
    # Try L-BFGS-B from the seed with tight tolerances
    if _time_left() > 20:
        try:
            r = minimize(_objective, best_x, method='L-BFGS-B', jac=_gradient,
                         options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16})
            if r.fun < best_v:
                best_v = r.fun
                best_x = r.x.copy()
        except Exception:
            pass
    
    # Try from a few alternative starting points
    alt_starts = [
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0002]),
        np.array([-0.9, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00031]),
        np.array([-1.15, 0.042, 0.00029]),
    ]
    for x0 in alt_starts:
        if _time_left() < 15:
            break
        try:
            r = minimize(_objective, x0, method='L-BFGS-B', jac=_gradient,
                         options={'maxiter': 10000, 'gtol': 1e-14, 'ftol': 1e-16})
            if r.fun < best_v:
                best_v = r.fun
                best_x = r.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if _time_left() > 5:
        try:
            r = minimize(_objective, best_x, method='L-BFGS-B', jac=_gradient,
                         options={'maxiter': 50000, 'gtol': 1e-16, 'ftol': 1e-18})
            if r.fun < best_v:
                best_v = r.fun
                best_x = r.x.copy()
        except Exception:
            pass
    
    return np.array([best_x[0], best_x[1], best_x[2]])
