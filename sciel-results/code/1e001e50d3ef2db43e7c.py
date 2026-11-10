import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    x, w = np.polynomial.legendre.leggauss(70)
    # x in [-1,1] is cos(theta), w are weights
    # phi equally spaced
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    # Build 2D grid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    # Surface measure: dOmega = dphi * d(cos_theta), so weights are w_i * (2*pi/130)
    # Actually the quadrature: integral = sum_i sum_j f(x_i, phi_j) * w_i * (2*pi/130)
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    # Compute P_10^m(cos_theta) with Condon-Shortley phase
    # scipy.special.lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)  # shape (70, 130)
    
    # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    # Build real orthonormal harmonics Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0
    # Y_m = sqrt(2) * N_m * P_10^m * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m| * sin(|m|*phi) for m < 0
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _objective(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (70, 130)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    return norm4 / norm2

def _objective_and_grad(coeffs, Y, weights_2d):
    """Compute objective and gradient."""
    f = np.tensordot(coeffs, Y, axes=(0, 0))
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    
    # d||f||4/dcoeffs = (1/4) * ||f||4^(-3) * 4 * sum(f^3 * Y_m * w)
    # d||f||2/dcoeffs = (1/2) * ||f||2^(-1) * 2 * sum(f * Y_m * w)
    # d(norm4/norm2)/dc = (1/norm2) * dnorm4/dc - (norm4/norm2^2) * dnorm2/dc
    
    f3Y = np.tensordot(f**3, Y, axes=(0, 0))
    fY = np.tensordot(f, Y, axes=(0, 0))
    
    dnorm4_dc = np.sum(f3Y * w[None, :, :] if f3Y.ndim == 2 else f3Y * w, axis=(-1, -2)) / (norm4**3)
    dnorm2_dc = np.sum(fY * w, axis=(-1, -2)) / norm2
    
    grad = dnorm4_dc / norm2 - (norm4 / norm2**2) * dnorm2_dc
    
    return norm4 / norm2, grad

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin
    
    # Precompute harmonics and weights
    Y, weights_2d = _build_harmonics()
    
    # Verify orthonormality
    # sum_m Y_m * Y_n * w = delta_mn
    # This is a sanity check
    
    # Initial guess: all ones
    best_coeffs = np.ones(21)
    best_score = _objective(best_coeffs, Y, weights_2d)
    
    # Strategy 1: Try some structured initial guesses
    # Zero out most, keep a few
    # Try: only m=0
    c0 = np.zeros(21)
    c0[10] = 1.0  # m=0
    s0 = _objective(c0, Y, weights_2d)
    if s0 < best_score:
        best_score = s0
        best_coeffs = c0.copy()
    
    # Try: m=0 and m=10
    c1 = np.zeros(21)
    c1[0] = 1.0   # m=-10
    c1[20] = 1.0  # m=10
    s1 = _objective(c1, Y, weights_2d)
    if s1 < best_score:
        best_score = s1
        best_coeffs = c1.copy()
    
    # Try: symmetric cos terms only
    c2 = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        if m == 0:
            c2[10] = 1.0
        else:
            c2[10+m] = 1.0  # m>0
    s2 = _objective(c2, Y, weights_2d)
    if s2 < best_score:
        best_score = s2
        best_coeffs = c2.copy()
    
    # Try: all even m cos terms
    c3 = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        if m == 0:
            c3[10] = 1.0
        else:
            c3[10+m] = 1.0
            c3[10-m] = 0.0  # don't add sin
    s3 = _objective(c3, Y, weights_2d)
    if s3 < best_score:
        best_score = s3
        best_coeffs = c3.copy()
    
    # Try random starts with local optimization
    np.random.seed(42)
    
    # Multiple random restarts with L-BFGS-B
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Random initial point
        x0 = np.random.randn(21)
        x0 /= np.linalg.norm(x0) * 10  # scale to reasonable magnitude
        
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                s = res.fun
                if s < best_score and np.isfinite(s):
                    best_score = s
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Differential evolution for global search
    if time.time() - start_time < deadline - 120:
        try:
            bounds = [(-10, 10)] * 21
            de_res = differential_evolution(
                lambda c: _objective(c, Y, weights_2d),
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False,
                init='sobol'
            )
            if de_res.fun < best_score and np.isfinite(de_res.fun):
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish with high-precision L-BFGS-B
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(
                lambda c: _objective(c, Y, weights_2d),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_coeffs *= 1.0 / max_abs if max_abs > 0 else 1.0
        elif max_abs > 1e5:
            best_coeffs *= 1e5 / max_abs
        if not np.all(np.isfinite(best_coeffs)):
            return np.ones(21)
    
    return best_coeffs.astype(np.float64)
