"""Spectral moment approach: minimize uncertainty bound via exact Gaussian quadrature
and multi-start gradient optimization."""

import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = time.time() + 160

# Pre-compute Gauss-Hermite nodes and weights for weight function exp(-x^2)
# We need enough nodes to integrate polynomials of degree up to 2*6=12 exactly?
# Gauss-Hermite with n nodes integrates polynomials of degree 2n-1 exactly.
# Our integrand is |P(x)|^2 x^{2k} e^{-x^2}.
# P(x) is degree 12 (max). |P(x)|^2 is degree 24.
# x^{2k} adds up to degree 4 (for k=2).
# Total degree is 24 + 4 = 28.
# We need 2n - 1 >= 28 => n >= 14.5 => n=15.
# Let's use 50 nodes to be safe and robust against any slight numerical drift or if we tweak degrees.
_X_GH, _W_GH = np.polynomial.hermite.hermgauss(50)

def _make_g_coefficients(a0, a4, a8):
    """
    Given free coefficients a0, a4, a8, compute all polynomial coefficients
    P(x) = sum_{k=0}^{6} p_k x^{2k} for g(x) = P(x) e^{-x^2/2}.
    a12 is fixed to enforce g(0) = 0.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    # Coefficients for powers x^0, x^2, x^4, ..., x^12
    # p0 * x^0 + p1 * x^2 + p2 * x^4 + ...
    return np.array([p0, p1, p2, p3, p4, p5, p6]), a12

def _evaluate_bound(params):
    """
    Compute the uncertainty bound for given parameters using Gauss-Hermite quadrature.
    """
    p, _ = _make_g_coefficients(params[0], params[1], params[2])
    
    # Evaluate P(x) at all GH nodes
    # P(x) = p0 + p1 x^2 + p2 x^4 + ...
    # x^2 is the variable for the polynomial in x^2
    x2 = _X_GH**2
    # Use Horner's rule or polyval on x^2
    # p is coeffs for x^0, x^2, ..., x^12
    P_vals = np.polyval(p, x2)
    
    # |P(x)|^2
    P_sq = P_vals**2
    
    # Weights for moments
    # M_k = sum w_i x_i^{2k} |P(x_i)|^2
    # k=0: sum w_i P_sq
    # k=1: sum w_i x_i^2 P_sq
    # k=2: sum w_i x_i^4 P_sq
    
    w = _W_GH
    x2_w = w * x2
    x4_w = w * x2 * x2
    
    M0 = np.sum(w * P_sq)
    M1 = np.sum(x2_w * P_sq)
    M2 = np.sum(x4_w * P_sq)
    
    if M0 <= 0 or not np.isfinite(M0):
        return 1e18
        
    # Fourier side
    # Q(xi) = P(-xi) ? No, Fourier transform of P(x)e^{-x^2/2} is related to P(-i xi) or similar.
    # For Physicists' Hermite, FT of H_n(x) e^{-x^2/2} is (-i)^n H_n(xi) e^{-xi^2/2} (up to constants).
    # Specifically, if g(x) = P(x) e^{-x^2/2}, then G(xi) = \hat{g}(xi) = \int P(x) e^{-x^2/2} e^{-i x xi} dx.
    # The seed code uses: q_k = p_k (-1)^k.
    # This implies Q(xi) = \sum p_k (-1)^k xi^{2k}.
    # Let's stick to the seed's logic for the Fourier side polynomial.
    
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    # Evaluate Q(xi) at GH nodes (xi distribution is same as x due to symmetry)
    Q_vals = np.polyval(q, x2)
    Q_sq = Q_vals**2
    
    # N_k = 2*pi * integral xi^{2k} |Q(xi)|^2 e^{-xi^2} dxi
    # Factor 2*pi is constant, doesn't affect argmin, but let's keep it for consistency with seed's definition of bound.
    # Actually, the bound is sqrt(Vf * VF).
    # Vf = M1/M0.
    # VF = N1/N0.
    # The factor 2*pi cancels in N1/N0.
    # So we can just compute N0, N1 from Q_sq.
    
    N0 = np.sum(w * Q_sq)
    N1 = np.sum(x2_w * Q_sq)
    
    if N0 <= 0 or not np.isfinite(N0):
        return 1e18
        
    Vf = M1 / M0
    VF = N1 / N0
    
    if Vf <= 0 or VF <= 0:
        return 1e18
        
    return np.sqrt(Vf * VF)

def construct():
    """
    Find optimal Hermite coefficients to minimize the uncertainty bound.
    Returns array of three coefficients [a0, a4, a8].
    """
    # Seed starting point
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(2024)
    starts = [seed]
    
    # Generate diverse starting points
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0]:
        for _ in range(10):
            starts.append(seed + rng.normal(0, scale, 3))
            
    # Random starts
    for _ in range(50):
        starts.append(rng.uniform(-5, 5, 3))
        
    best_val = _evaluate_bound(seed)
    best_params = seed.copy()
    
    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            # L-BFGS-B is robust and fast
            res = minimize(
                _evaluate_bound, start,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    # Final polish
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(
                _evaluate_bound, best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None or not np.all(np.isfinite(best_params)):
        return seed
        
    return np.array(best_params)
