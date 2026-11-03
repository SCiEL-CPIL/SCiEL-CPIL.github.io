"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h_prev = np.ones_like(x)
    h_curr = 2 * x
    for k in range(2, n + 1):
        h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
        h_prev = h_curr
        h_curr = h_next
    return h_curr

def _compute_bound(coeffs):
    """
    Compute the upper bound C4 given coefficients [a0, a4, a8].
    a12 is determined by g(0) = 0.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/4)
    
    The bound involves the L2 norm of g and its Fourier transform.
    For Hermite-Gaussian functions, we can use orthogonality.
    """
    a0, a4, a8 = coeffs
    
    # Compute a12 from g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1008, H_12(0)=-103216
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1008.0
    H12_0 = -103216.0
    
    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * exp(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs_all = np.array([a0, a4, a8, a12])
    degrees = [0, 4, 8, 12]
    
    # The L2 norm of g(x) = P(x) * exp(-x^2/4)
    # ||g||_2^2 = integral of P(x)^2 * exp(-x^2/2) dx
    # Using orthogonality: integral H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_mn
    # But we have exp(-x^2/2), not exp(-x^2). Let me rescale.
    # 
    # Actually, let's use: integral H_m(x) H_n(x) exp(-x^2/2) dx
    # With the substitution y = x/sqrt(2), dx = sqrt(2) dy, exp(-x^2/2) = exp(-y^2)
    # integral H_m(sqrt(2)y) H_n(sqrt(2)y) exp(-y^2) sqrt(2) dy
    #
    # This is getting complex. Let me use numerical integration instead for reliability.
    
    # Use numerical quadrature
    x = np.linspace(-20, 20, 100000)
    dx = x[1] - x[0]
    
    # Compute P(x)
    P = np.zeros_like(x)
    for a, d in zip(coeffs_all, degrees):
        P += a * _hermite(d, x)
    
    g = P * np.exp(-x**2 / 4)
    
    # ||g||_2^2
    norm_g_sq = np.sum(g**2) * dx
    
    # Fourier transform of g: G(xi) = integral g(x) exp(-2*pi*i*x*xi) dx
    # For the bound, we need ||G||_2^2 = integral |G(xi)|^2 dxi
    # By Parseval: ||G||_2^2 = ||g||_2^2 (for the unitary FT)
    # But the standard FT has different normalizations.
    #
    # For F(f)(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # Parseval: integral |f(x)|^2 dx = integral |F(f)(xi)|^2 dxi
    # So ||G||_2^2 = ||g||_2^2
    #
    # The uncertainty product bound is related to the spread of g and G.
    # For the specific C4 bound in the uncertainty inequality context,
    # it's typically: C4 = (integral x^2 |g(x)|^2 dx) * (integral xi^2 |G(xi)|^2 dxi)
    # or a similar product of second moments.
    
    # Compute second moment of g
    x2_moment_g = np.sum(x**2 * g**2) * dx
    
    # For the Fourier side, we need the second moment of |G(xi)|^2
    # By symmetry and the properties of the FT, for even functions:
    # integral xi^2 |G(xi)|^2 dxi = (1/(4*pi^2)) * integral |g''(x)|^2 dx
    # (using the property that FT of g'' is (2*pi*i*xi)^2 G(xi))
    
    # Compute g''(x)
    # g(x) = P(x) * exp(-x^2/4)
    # g'(x) = P'(x)*exp(-x^2/4) + P(x)*(-x/2)*exp(-x^2/4)
    # g''(x) = P''(x)*exp(-x^2/4) + P'(x)*(-x/2)*exp(-x^2/4) + P'(x)*(-x/2)*exp(-x^2/4) + P(x)*(1/4 - x^2/4)*exp(-x^2/4)
    # = [P''(x) - x*P'(x) + P(x)*(1/4 - x^2/4)] * exp(-x^2/4)
    
    # Compute P'(x) and P''(x)
    P_prime = np.zeros_like(x)
    P_double_prime = np.zeros_like(x)
    for a, d in zip(coeffs_all, degrees):
        # H_n'(x) = 2n * H_{n-1}(x)
        if d >= 1:
            P_prime += a * 2 * d * _hermite(d - 1, x)
        # H_n''(x) = 2n * (2n-2) * H_{n-2}(x) for n >= 2
        if d >= 2:
            P_double_prime += a * 2 * d * (2 * d - 2) * _hermite(d - 2, x)
    
    g_double_prime = (P_double_prime - x * P_prime + P * (1/4 - x**2 / 4)) * np.exp(-x**2 / 4)
    
    norm_gpp_sq = np.sum(g_double_prime**2) * dx
    
    # integral xi^2 |G(xi)|^2 dxi = (1/(4*pi^2)) * ||g''||_2^2
    xi2_moment_G = norm_gpp_sq / (4 * np.pi**2)
    
    # The bound C4 is the product of the second moments
    C4 = x2_moment_g * xi2_moment_G
    
    return C4

def construct():
    """
    Find coefficients [a0, a4, a8] that minimize the upper bound C4.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    deadline = start_time + 180.0  # 180 second limit
    
    # Initial guess: the seed values
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    best_x = x0.copy()
    best_bound = _compute_bound(x0)
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.05, 0.042, 0.00032]),
        np.array([-1.15, 0.048, 0.00028]),
        np.array([-0.95, 0.038, 0.00038]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _compute_bound,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Try Nelder-Mead as a backup
    if time.time() < deadline - 60:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tight tolerances
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _compute_bound,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)

# EVOLVE-BLOCK-END
