"""Optimized program: Spherical reparameterization + high-precision quadrature + analytic gradient."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 10.0

# Precompute grid and polynomial values once
_NPTS = 4001
_X = np.linspace(-12.0, 12.0, _NPTS)
_DX = _X[1] - _X[0]
_GAUSS = np.exp(-0.5 * _X * _X)

# Precompute Hermite polynomials and their derivatives at grid points
def _H(n, x):
    """Physicist's Hermite polynomial H_n(x) via explicit sum."""
    N = int(n)
    out = np.zeros_like(x, dtype=float)
    for k in range(N // 2 + 1):
        log_coef = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - 2 * k + 1)
        coef = np.exp(log_coef) * ((-1.0) ** k)
        out += coef * (2.0 * x) ** (N - 2 * k)
    return out

_H0 = np.ones_like(_X)
_H4 = _H(4, _X)
_H8 = _H(8, _X)
_H12 = _H(12, _X)
_H3 = _H(3, _X)
_H7 = _H(7, _X)
_H11 = _H(11, _X)

# Derivatives: H_n'(x) = 2n H_{n-1}(x)
_dH0 = np.zeros_like(_X)
_dH4 = 8.0 * _H3
_dH8 = 16.0 * _H7
_dH12 = 24.0 * _H11

# Precompute FFT setup
_N = _NPTS
_OMEGA = 2.0 * np.pi * np.fft.fftfreq(_N, d=_DX)
_OMEGA_SORTED = np.sort(_OMEGA)
_OMEGA_IDX = np.argsort(_OMEGA)

# Values at x=0 for the constraint
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 168.0
_H12_0 = -4960.0

# Spherical reparameterization:
# The constraint g(0) = 0 means a0 + a4*(-12) + a8*168 + a12*(-4960) = 0
# => a12 = (a0 - 12*a4 + 168*a8) / 4960
# We parametrize (a0, a4, a8) on a 2D manifold.
# Use spherical coordinates: a0 = r*cos(theta)*cos(phi), a4 = r*cos(theta)*sin(phi), a8 = r*sin(theta)
# This makes the search bounded in (theta, phi) with r as scale.
# But since the objective may have scale invariance, we fix r and search over (theta, phi).

# Precompute the "basis" vectors for the constrained subspace
# Given (u, v) in 2D, map to (a0, a4, a8) via:
# a0 = u, a4 = v, a8 = w where w is determined by another parameter
# Actually, let's use: a0 = r*cos(theta), a4 = r*sin(theta)*cos(phi), a8 = r*sin(theta)*sin(phi)
# This gives a 2D angular parameterization (theta, phi) with scale r.


def _build_g_and_grad_components(a0, a4, a8):
    """Return g, gp, and the polynomial components for gradient computation."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    p = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    dp = a0 * _dH0 + a4 * _dH4 + a8 * _dH8 + a12 * _dH12
    
    g = p * _GAUSS
    gp = (dp - _X * p) * _GAUSS
    
    return g, gp, a12


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound value and its gradient."""
    g, gp, a12 = _build_g_and_grad_components(a0, a4, a8)
    
    # Fourier transform via FFT
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * _DX
    G_sorted = G[_OMEGA_IDX]
    
    # Interpolate G onto the x-grid (even function, so real part)
    Gx = np.interp(_X, _OMEGA_SORTED, G_sorted.real)
    
    # Integrals using Simpson's rule for better accuracy
    I_g = np.trapz(g, _X)
    I_gp2 = np.trapz(gp ** 2, _X)
    I_G2 = np.trapz(Gx ** 2, _X)
    
    # Bound: |int g| * sqrt(int (g')^2) * sqrt(int |G|^2)
    abs_Ig = abs(I_g)
    sqrt_Igp2 = np.sqrt(max(I_gp2, 0.0))
    sqrt_IG2 = np.sqrt(max(I_G2, 0.0))
    
    val = abs_Ig * sqrt_Igp2 * sqrt_IG2
    
    # Analytic gradient via chain rule
    sign_Ig = np.sign(I_g) if I_g != 0 else 1.0
    
    da12_da0 = _H0_0 / _H12_0
    da12_da4 = _H4_0 / _H12_0
    da12_da8 = _H8_0 / _H12_0
    
    dIg_da0 = np.trapz((_H0 + da12_da0 * _H12) * _GAUSS, _X)
    dIg_da4 = np.trapz((_H4 + da12_da4 * _H12) * _GAUSS, _X)
    dIg_da8 = np.trapz((_H8 + da12_da8 * _H12) * _GAUSS, _X)
    
    dgp_da0 = (_dH0 + da12_da0 * _dH12 - _X * (_H0 + da12_da0 * _H12)) * _GAUSS
    dgp_da4 = (_dH4 + da12_da4 * _dH12 - _X * (_H4 + da12_da4 * _H12)) * _GAUSS
    dgp_da8 = (_dH8 + da12_da8 * _dH12 - _X * (_H8 + da12_da8 * _H12)) * _GAUSS
    
    dIgp2_da0 = 2.0 * np.trapz(gp * dgp_da0, _X)
    dIgp2_da4 = 2.0 * np.trapz(gp * dgp_da4, _X)
    dIgp2_da8 = 2.0 * np.trapz(gp * dgp_da8, _X)
    
    dg_da0 = (_H0 + da12_da0 * _H12) * _GAUSS
    dg_da4 = (_H4 + da12_da4 * _H12) * _GAUSS
    dg_da8 = (_H8 + da12_da8 * _H12) * _GAUSS
    
    Gg_da0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(dg_da0))) * _DX
    Gg_da4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(dg_da4))) * _DX
    Gg_da8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(dg_da8))) * _DX
    
    Gg_da0_sorted = Gg_da0[_OMEGA_IDX]
    Gg_da4_sorted = Gg_da4[_OMEGA_IDX]
    Gg_da8_sorted = Gg_da8[_OMEGA_IDX]
    
    dGx_da0 = np.interp(_X, _OMEGA_SORTED, Gg_da0_sorted.real)
    dGx_da4 = np.interp(_X, _OMEGA_SORTED, Gg_da4_sorted.real)
    dGx_da8 = np.interp(_X, _OMEGA_SORTED, Gg_da8_sorted.real)
    
    dIG2_da0 = 2.0 * np.trapz(Gx * dGx_da0, _X)
    dIG2_da4 = 2.0 * np.trapz(Gx * dGx_da4, _X)
    dIG2_da8 = 2.0 * np.trapz(Gx * dGx_da8, _X)
    
    if sqrt_Igp2 > 1e-12 and sqrt_IG2 > 1e-12:
        grad0 = sign_Ig * (dIg_da0 * sqrt_Igp2 * sqrt_IG2 
                         + I_g * dIgp2_da0 / (2.0 * sqrt_Igp2) * sqrt_IG2
                         + I_g * sqrt_Igp2 * dIG2_da0 / (2.0 * sqrt_IG2))
        grad4 = sign_Ig * (dIg_da4 * sqrt_Igp2 * sqrt_IG2 
                         + I_g * dIgp2_da4 / (2.0 * sqrt_Igp2) * sqrt_IG2
                         + I_g * sqrt_Igp2 * dIG2_da4 / (2.0 * sqrt_IG2))
        grad8 = sign_Ig * (dIg_da8 * sqrt_Igp2 * sqrt_IG2 
                         + I_g * dIgp2_da8 / (2.0 * sqrt_Igp2) * sqrt_IG2
                         + I_g * sqrt_Igp2 * dIG2_da8 / (2.0 * sqrt_IG2))
    else:
        grad0 = grad4 = grad8 = 0.0
    
    return val, np.array([grad0, grad4, grad8])


def _objective(coeffs):
    """Objective function for minimization."""
    val, _ = _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    return val


def _objective_and_grad(coeffs):
    """Return (value, gradient) for L-BFGS-B."""
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])


def _spherical_to_coeffs(theta, phi, r):
    """Convert spherical coordinates to (a0, a4, a8)."""
    a0 = r * np.cos(theta)
    a4 = r * np.sin(theta) * np.cos(phi)
    a8 = r * np.sin(theta) * np.sin(phi)
    return np.array([a0, a4, a8])


def _coeffs_to_spherical(a0, a4, a8):
    """Convert (a0, a4, a8) to spherical coordinates (theta, phi, r)."""
    r = np.sqrt(a0**2 + a4**2 + a8**2)
    if r < 1e-15:
        return 0.0, 0.0, r
    theta = np.arccos(np.clip(a0 / r, -1.0, 1.0))
    phi = np.arctan2(a8, a4)
    return theta, phi, r


def _objective_spherical(params):
    """Objective in spherical coordinates (theta, phi, r)."""
    theta, phi, r = params
    coeffs = _spherical_to_coeffs(theta, phi, r)
    val, _ = _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    return val


def _objective_spherical_and_grad(params):
    """Return (value, gradient) in spherical coordinates."""
    theta, phi, r = params
    coeffs = _spherical_to_coeffs(theta, phi, r)
    val, grad_coeffs = _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    # Chain rule: d(val)/d(theta), d(val)/d(phi), d(val)/d(r)
    # a0 = r*cos(theta), a4 = r*sin(theta)*cos(phi), a8 = r*sin(theta)*sin(phi)
    da0_dtheta = -r * np.sin(theta)
    da4_dtheta = r * np.cos(theta) * np.cos(phi)
    da8_dtheta = r * np.cos(theta) * np.sin(phi)
    
    da0_dphi = 0.0
    da4_dphi = -r * np.sin(theta) * np.sin(phi)
    da8_dphi = r * np.sin(theta) * np.cos(phi)
    
    da0_dr = np.cos(theta)
    da4_dr = np.sin(theta) * np.cos(phi)
    da8_dr = np.sin(theta) * np.sin(phi)
    
    g0, g4, g8 = grad_coeffs
    grad_theta = g0 * da0_dtheta + g4 * da4_dtheta + g8 * da8_dtheta
    grad_phi = g0 * da0_dphi + g4 * da4_dphi + g8 * da8_dphi
    grad_r = g0 * da0_dr + g4 * da4_dr + g8 * da8_dr
    
    return val, np.array([grad_theta, grad_phi, grad_r])


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point.
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0, _ = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Multi-start L-BFGS-B in Cartesian coordinates
    rng = np.random.default_rng(42)
    
    starts = [best_coeffs.copy()]
    for _ in range(10):
        starts.append(best_coeffs + rng.normal(0, 0.05, size=3))
    
    for scale in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        starts.append(best_coeffs * scale)
    
    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-13, 'gtol': 1e-11},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Spherical coordinate optimization for additional exploration
    if time.time() < _DEADLINE - _MARGIN:
        theta0, phi0, r0 = _coeffs_to_spherical(best_coeffs[0], best_coeffs[1], best_coeffs[2])
        spherical_starts = [
            np.array([theta0, phi0, r0]),
            np.array([theta0 + 0.3, phi0, r0]),
            np.array([theta0 - 0.3, phi0, r0]),
            np.array([theta0, phi0 + 0.5, r0]),
            np.array([theta0, phi0 - 0.5, r0]),
            np.array([theta0, phi0, r0 * 0.5]),
            np.array([theta0, phi0, r0 * 2.0]),
            np.array([0.5, 0.5, r0]),
            np.array([1.0, 1.0, r0]),
            np.array([1.5, 0.3, r0]),
        ]
        
        for x0 in spherical_starts:
            if time.time() > _DEADLINE - _MARGIN:
                break
            try:
                res = minimize(
                    _objective_spherical_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (1e-6, 10.0)],
                    options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-13, 'gtol': 1e-11},
                )
                if res.fun < best_val:
                    new_coeffs = _spherical_to_coeffs(res.x[0], res.x[1], res.x[2])
                    if np.all(np.isfinite(new_coeffs)):
                        best_val = res.fun
                        best_coeffs = new_coeffs
            except Exception:
                continue
    
    # Final high-precision polish in Cartesian
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    return best_coeffs
# EVOLVE-BLOCK-END
