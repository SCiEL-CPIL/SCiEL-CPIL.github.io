"""Optimized program: Remez exchange + analytic gradient L-BFGS-B with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 10.0

# Precompute grid and polynomial values once
_NPTS = 3001
_X = np.linspace(-10.0, 10.0, _NPTS)
_DX = _X[1] - _X[0]
_GAUSS = np.exp(-0.5 * _X * _X)

# Precompute Hermite polynomials and their derivatives at grid points
def _H(n, x):
    """Physicist's Hermite polynomial H_n(x) via explicit sum."""
    N = int(n)
    out = np.zeros_like(x, dtype=float)
    for k in range(N // 2 + 1):
        # Use log-gamma for numerical stability
        from scipy.special import gammaln
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
    
    # Integrals
    I_g = np.trapz(g, _X)
    I_gp2 = np.trapz(gp ** 2, _X)
    I_G2 = np.trapz(Gx ** 2, _X)
    
    # Bound: |int g| * sqrt(int (g')^2) * sqrt(int |G|^2)
    abs_Ig = abs(I_g)
    sqrt_Igp2 = np.sqrt(max(I_gp2, 0.0))
    sqrt_IG2 = np.sqrt(max(I_G2, 0.0))
    
    val = abs_Ig * sqrt_Igp2 * sqrt_IG2
    
    # Analytic gradient via chain rule
    # d(val)/d(a_i) = sign(I_g) * [d(I_g)/d(a_i) * sqrt_Igp2 * sqrt_IG2
    #                            + I_g * (1/(2*sqrt_Igp2)) * d(I_gp2)/d(a_i) * sqrt_IG2
    #                            + I_g * sqrt_Igp2 * (1/(2*sqrt_IG2)) * d(I_G2)/d(a_i)]
    
    sign_Ig = np.sign(I_g) if I_g != 0 else 1.0
    
    # d(I_g)/d(a_i) = int dp_i * e^{-x^2/2} dx where dp_i is derivative of p wrt a_i
    # p = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # a12 = -(a0*1 + a4*(-12) + a8*168)/(-4960) = (a0 + 12*a4 - 168*a8)/4960
    
    da12_da0 = _H0_0 / _H12_0  # 1/(-4960)
    da12_da4 = _H4_0 / _H12_0  # -12/(-4960)
    da12_da8 = _H8_0 / _H12_0  # 168/(-4960)
    
    # dI_g/da0 = int (H0 + da12_da0*H12) * gauss * dx
    dIg_da0 = np.trapz((_H0 + da12_da0 * _H12) * _GAUSS, _X)
    dIg_da4 = np.trapz((_H4 + da12_da4 * _H12) * _GAUSS, _X)
    dIg_da8 = np.trapz((_H8 + da12_da8 * _H12) * _GAUSS, _X)
    
    # dI_gp2/da_i: gp = (dp - x*p)*gauss
    # dp = a0*dH0 + a4*dH4 + a8*dH8 + a12*dH12
    # d(gp)/da0 = (dH0 + da12_da0*dH12 - x*(H0 + da12_da0*H12)) * gauss
    dgp_da0 = (_dH0 + da12_da0 * _dH12 - _X * (_H0 + da12_da0 * _H12)) * _GAUSS
    dgp_da4 = (_dH4 + da12_da4 * _dH12 - _X * (_H4 + da12_da4 * _H12)) * _GAUSS
    dgp_da8 = (_dH8 + da12_da8 * _dH12 - _X * (_H8 + da12_da8 * _H12)) * _GAUSS
    
    dIgp2_da0 = 2.0 * np.trapz(gp * dgp_da0, _X)
    dIgp2_da4 = 2.0 * np.trapz(gp * dgp_da4, _X)
    dIgp2_da8 = 2.0 * np.trapz(gp * dgp_da8, _X)
    
    # dI_G2/da_i: G = FFT(g), so dG/da_i = FFT(dg/da_i)
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
    
    # Assemble gradient
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


def _remez_exchange(coeffs, max_iter=20):
    """Remez-style exchange: find extremal points and adjust coefficients."""
    best_coeffs = coeffs.copy()
    best_val = _objective(best_coeffs)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Sample the bound landscape on a fine grid of coefficient perturbations
        # to find where the bound is maximized (worst case)
        n_samples = 50
        rng = np.random.default_rng(it)
        perturbations = rng.normal(0, 0.02, size=(n_samples, 3))
        
        vals = []
        for p in perturbations:
            try:
                v = _objective(best_coeffs + p)
                vals.append(v)
            except Exception:
                vals.append(np.inf)
        
        vals = np.array(vals)
        
        # Find the direction that increases the bound most (anti-gradient step)
        idx_max = np.argmax(vals)
        if not np.isfinite(vals[idx_max]):
            break
        
        # Move toward the extremal point
        step = perturbations[idx_max]
        # Damped step
        alpha = 0.5
        new_coeffs = best_coeffs + alpha * step
        
        new_val = _objective(new_coeffs)
        if new_val < best_val:
            best_val = new_val
            best_coeffs = new_coeffs
    
    return best_coeffs, best_val


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point.
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0, _ = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Multi-start L-BFGS-B with proper analytic gradient
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [best_coeffs.copy()]
    for _ in range(12):
        starts.append(best_coeffs + rng.normal(0, 0.1, size=3))
    
    # Also add some structured starts
    for scale in [0.5, 1.0, 2.0]:
        starts.append(np.array([-113.0/100 * scale, 1.0/25 * scale, 1.0/3240 * scale]))
    
    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Remez exchange polish on the best solution found
    if time.time() < _DEADLINE - _MARGIN:
        try:
            remez_coeffs, remez_val = _remez_exchange(best_coeffs, max_iter=30)
            if remez_val < best_val:
                best_val = remez_val
                best_coeffs = remez_coeffs
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - _MARGIN:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12},
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
