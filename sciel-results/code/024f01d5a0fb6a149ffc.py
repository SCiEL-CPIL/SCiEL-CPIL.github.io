"""Hermite-Gaussian tuning for the L4 uncertainty constant C4.

We minimize the test-function upper bound
    C4_upper = ||f||_4 * ||F[f]||_4 / ||f||_2
for f(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2},
with c12 fixed by g(0)=0.  Lower is better.

Structural change: represent f in the orthonormal Hermite-function basis
psi_k(x) = (2^k k! sqrt(pi))^{-1/2} H_k(x) e^{-x^2/2}.
Then ||f||_2^2 = sum a_k^2 exactly (by orthonormality), and the Fourier
transform is also a sum of Hermite functions (up to a phase), so
||F[f]||_2^2 = ||f||_2^2.  The L4 norms are computed via a fine quadrature
on the basis representation, which is far more stable than FFT of a
truncated signal.  The objective is a smooth ratio of polynomial-in-a
forms; we minimize it with a global + local optimizer.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 180.0 - 30.0  # safety margin below the 210s kill / 180s limit

# Degrees of the Hermite polynomials we use
_DEG = (0, 4, 8, 12)
_NDEG = len(_DEG)

# Normalization for orthonormal Hermite functions:
#   psi_k(x) = (2^k k! sqrt(pi))^{-1/2} H_k(x) e^{-x^2/2}
# so that <psi_k, psi_l> = delta_kl.
# We work with f(x) = sum_k a_k psi_k(x).
# The constraint g(0)=0 means sum_k a_k psi_k(0) = 0.
# psi_k(0) = 0 for odd k; for even k = 2m:
#   psi_{2m}(0) = (2^{2m} (2m)! sqrt(pi))^{-1/2} H_{2m}(0)
#   H_{2m}(0) = (-1)^m (2m)! / m!
# so psi_{2m}(0) = (-1)^m (2m)! / (m! sqrt(2^{2m} (2m)! sqrt(pi)))
#                = (-1)^m sqrt((2m)!) / (m! sqrt(2^{2m} sqrt(pi)))
#                = (-1)^m sqrt((2m)! / (m!^2 2^{2m} sqrt(pi)))

def _psi_at_0(k):
    """Value of the orthonormal Hermite function psi_k at x=0."""
    if k % 2 != 0:
        return 0.0
    m = k // 2
    # psi_{2m}(0) = (-1)^m * sqrt((2m)!) / (m! * 2^m * pi^{1/4})
    # Derivation: H_{2m}(0) = (-1)^m (2m)!/m!
    # psi_{2m}(0) = H_{2m}(0) / sqrt(2^{2m} (2m)! sqrt(pi))
    #             = (-1)^m (2m)!/m! / sqrt(2^{2m} (2m)! pi^{1/2})
    #             = (-1)^m sqrt((2m)!) / (m! 2^m pi^{1/4})
    sign = 1.0 if m % 2 == 0 else -1.0
    val = sign * np.sqrt(factorial(2 * m, exact=True)) / (
        float(factorial(m, exact=True)) * (2.0 ** m) * (np.pi ** 0.25))
    return val

# Values of psi_k at x=0 for our degrees
_PSI0 = np.array([_psi_at_0(k) for k in _DEG])

# Quadrature grid for computing L4 norms
_XQ = np.linspace(-12.0, 12.0, 16384)
_DXQ = _XQ[1] - _XQ[0]
_GAUSS_Q = np.exp(-0.5 * _XQ * _XQ)
# psi_k(x) on the grid
_PSI_GRID = np.zeros((_NDEG, len(_XQ)))
for i, k in enumerate(_DEG):
    norm = np.sqrt(2.0**k * factorial(k, exact=True) * np.sqrt(np.pi))
    _PSI_GRID[i] = eval_hermite(k, _XQ) * _GAUSS_Q / norm

def _full_coeffs(c):
    """Given (c0, c4, c8) in the H_k e^{-x^2/2} basis, return a in the psi_k basis
    with the constraint sum a_k psi_k(0) = 0 enforced by solving for a_12."""
    c0, c4, c8 = c
    # Convert from H_k e^{-x^2/2} coefficients to psi_k coefficients:
    # f = sum c_k H_k e^{-x^2/2} = sum a_k psi_k
    # a_k = c_k * sqrt(2^k k! sqrt(pi))
    scales = np.array([np.sqrt(2.0**k * factorial(k, exact=True) * np.sqrt(np.pi)) for k in _DEG])
    a = np.zeros(_NDEG)
    a[0] = c0 * scales[0]
    a[1] = c4 * scales[1]
    a[2] = c8 * scales[2]
    # Constraint: a @ PSI0 = 0  =>  a[3] = -(a[0]*PSI0[0] + a[1]*PSI0[1] + a[2]*PSI0[2]) / PSI0[3]
    if abs(_PSI0[3]) < 1e-15:
        return a  # shouldn't happen
    a[3] = -(a[0] * _PSI0[0] + a[1] * _PSI0[1] + a[2] * _PSI0[2]) / _PSI0[3]
    return a

def _compute_objective(c):
    """Compute C4_upper = ||f||_4 * ||F[f]||_4 / ||f||_2 for f in psi basis."""
    a = _full_coeffs(c)
    
    # ||f||_2^2 = sum a_k^2 (orthonormality)
    n2_sq = np.dot(a, a)
    if n2_sq <= 0 or not np.isfinite(n2_sq):
        return 1e300
    n2 = np.sqrt(n2_sq)
    
    # f(x) on the grid
    f = _PSI_GRID @ a  # (N,)
    
    # ||f||_4^4 = ∫ |f|^4 dx
    n4_4 = np.sum(f**4) * _DXQ
    if n4_4 <= 0 or not np.isfinite(n4_4):
        return 1e300
    n4 = n4_4 ** 0.25
    
    # Fourier transform: F[psi_k](xi) = (-i)^k psi_k(xi) (for the convention
    # F[f](xi) = (1/sqrt(2pi)) ∫ f(x) e^{-ix xi} dx, which preserves L2 norm).
    # So F[f] = sum a_k (-i)^k psi_k(xi), and |F[f](xi)|^2 = |sum a_k (-i)^k psi_k(xi)|^2
    # Since |(-i)^k| = 1, |F[f](xi)| = |sum a_k e^{-i k pi/2} psi_k(xi)|
    # For L4: |F[f]|^4 = (sum a_k (-i)^k psi_k)^4... but we need |.|^4.
    # |F[f]|^2 = (sum a_k (-i)^k psi_k)(sum a_l i^l psi_l) = sum_{k,l} a_k a_l (-i)^k i^l psi_k psi_l
    # |F[f]|^4 = (|F[f]|^2)^2
    # Compute F_f on the grid:
    phases = np.array([(-1j)**k for k in _DEG])
    F_f = (_PSI_GRID * phases) @ a  # (N,)
    
    n4_freq_4 = np.sum(np.abs(F_f)**4) * _DXQ
    if n4_freq_4 <= 0 or not np.isfinite(n4_freq_4):
        return 1e300
    n4_freq = n4_freq_4 ** 0.25
    
    C4 = (n4 * n4_freq) / n2
    return C4

def _safe_objective(c):
    try:
        v = _compute_objective(c)
        if not np.isfinite(v):
            return 1e300
        return v
    except Exception:
        return 1e300

def construct():
    t_start = time.time()
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_v = _safe_objective(best_c)
    
    def remaining():
        return _DEADLINE - (time.time() - t_start)
    
    # 1) Global search with differential evolution (more iterations since objective is cheaper)
    if remaining() > 40:
        try:
            bounds = [(-3.0, 3.0), (-1.0, 1.0), (-0.2, 0.2)]
            res = differential_evolution(
                _safe_objective, bounds,
                seed=42, maxiter=150, popsize=20,
                tol=1e-12, polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res.x is not None and np.isfinite(res.fun) and res.fun < best_v:
                best_v = float(res.fun)
                best_c = np.asarray(res.x, dtype=float)
        except Exception:
            pass
    
    # 2) Local polish from multiple starts
    starts = [best_c.copy(),
              np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]),
              np.array([-1.0, 0.05, 0.001]),
              np.array([-1.5, 0.08, 0.0005]),
              np.array([-0.5, 0.02, 0.002]),
              np.array([-2.0, 0.1, 0.0003]),
              np.array([0.0, 0.0, 0.0]),
              np.array([-0.8, 0.04, 0.0015])]
    
    for s in starts:
        if remaining() < 10:
            break
        try:
            r = minimize(_safe_objective, s, method='Nelder-Mead',
                         options={'xatol': 1e-13, 'fatol': 1e-13,
                                  'maxiter': 800, 'adaptive': True})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass
    
    # 3) Final fine polish
    if remaining() > 5:
        try:
            r = minimize(_safe_objective, best_c, method='Nelder-Mead',
                         options={'xatol': 1e-15, 'fatol': 1e-15,
                                  'maxiter': 2000, 'adaptive': True})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
