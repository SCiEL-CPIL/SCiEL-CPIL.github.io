"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time


def _hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list, highest degree first."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [2.0, 0.0]
    if n == 2:
        return [4.0, 0.0, -2.0]
    if n == 3:
        return [8.0, 0.0, -12.0, 0.0]
    if n == 4:
        return [16.0, 0.0, -48.0, 0.0, 12.0]
    if n == 5:
        return [32.0, 0.0, -160.0, 0.0, 80.0, 0.0]
    if n == 6:
        return [64.0, 0.0, -480.0, 0.0, 720.0, 0.0, -60.0]
    if n == 7:
        return [128.0, 0.0, -1280.0, 0.0, 3360.0, 0.0, -1680.0, 0.0]
    if n == 8:
        return [256.0, 0.0, -3200.0, 0.0, 13440.0, 0.0, -13440.0, 0.0, 560.0]
    if n == 12:
        # H_12(x) = 4096x^12 - 122880x^10 + 1548288x^8 - 8808000x^6 + 20901120x^4 - 20901120x^2 + 7257600
        return [4096.0, 0.0, -122880.0, 0.0, 1548288.0, 0.0, -8808000.0, 0.0, 20901120.0, 0.0, -20901120.0, 0.0, 7257600.0]
    raise ValueError(f"n={n} not supported")


def _poly_mul(p1, p2):
    """Multiply two polynomials given as coefficient lists (highest degree first)."""
    n1, n2 = len(p1), len(p2)
    result = [0.0] * (n1 + n2 - 1)
    for i in range(n1):
        for j in range(n2):
            result[i + j] += p1[i] * p2[j]
    return result


def _poly_add(p1, p2):
    """Add two polynomials given as coefficient lists (highest degree first)."""
    n = max(len(p1), len(p2))
    p1_padded = [0.0] * (n - len(p1)) + list(p1)
    p2_padded = [0.0] * (n - len(p2)) + list(p2)
    return [a + b for a, b in zip(p1_padded, p2_padded)]


def _poly_scale(p, s):
    """Scale a polynomial by scalar s."""
    return [s * c for c in p]


def _compute_C4(c0, c1, c2):
    """
    Compute the upper bound C4 on A(f)*A(Fourier(f)) for
    g(x) = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * exp(-x^2/2)
    where c3 is chosen so g(0) = 0.
    
    The uncertainty product involves the L2 norm of certain cross-terms.
    For a function g(x) = P(x) * exp(-x^2/2), the relevant quantities are:
    - A(g) involves ||x * g||_2 and ||g'||_2 type norms
    - The product A(g)*A(Fourier(g)) can be bounded by C4 * ||g||_2^2
    
    Using the structure: g = P(x) e^{-x^2/2}, Fourier(g) = (i)^n-related terms.
    The key is computing the "uncertainty" which relates to the variance-like quantity.
    
    For the Hermite basis, the L2 norm of H_n(x) e^{-x^2/2} is sqrt(2^n n! sqrt(pi)).
    Cross terms between different H_n vanish due to orthogonality.
    
    The upper bound C4 comes from the product of the position uncertainty and momentum uncertainty.
    For g = P e^{-x^2/2}:
    - ||g||_2^2 = sum of |c_n|^2 * 2^n * n! * sqrt(pi) (orthogonality)
    - The position variance involves <x^2> and <x>^2
    - The momentum variance involves <p^2> and <p>^2 where p = -i d/dx
    
    The product A(g)*A(Fourier(g)) is bounded by C4 where C4 depends on the 
    ratio of these moments.
    """
    # Compute c3 from g(0) = 0
    # g(0) = (c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0)) * 1
    # H0(0) = 1, H4(0) = 12, H8(0) = -560, H12(0) = 7257600
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = -560.0
    H12_0 = 7257600.0
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Now compute the uncertainty product bound.
    # For g(x) = P(x) e^{-x^2/2}, the Fourier transform is also a Hermite-Gaussian.
    # The product A(g) * A(hat_g) where A is the "area" or L2 norm related quantity.
    # 
    # Actually, re-reading the task: "upper bound on A(f)*A(Fourier(f))"
    # A(f) likely means the L2 norm ||f||_2.
    # So we want to minimize ||g||_2 * ||hat_g||_2, but by Plancherel ||hat_g||_2 = ||g||_2,
    # so this is just ||g||_2^2. But that can be made arbitrarily small by scaling.
    # 
    # So A(f) must be something else - likely the "uncertainty" measure.
    # In uncertainty principles, A(f) might be the "spread" or standard deviation.
    # 
    # The standard uncertainty principle: sigma_x * sigma_p >= 1/2
    # The "upper bound" C4 is the best constant such that sigma_x * sigma_p <= C4 * ||g||_2^2
    # or similar.
    #
    # Given the context of Hermite polynomials and the specific form, let me compute
    # the actual variance-like quantities.
    
    # P(x) = c0*H0(x) + c1*H4(x) + c2*H8(x) + c3*H12(x)
    H0 = _hermite_coeffs(0)
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)
    
    P = _poly_add(_poly_add(_poly_add(
        _poly_scale(H0, c0),
        _poly_scale(H4, c1)
    ), _poly_scale(H8, c2)), _poly_scale(H12, c3))
    
    # P(x) e^{-x^2/2}
    # ||g||_2^2 = integral |P(x)|^2 e^{-x^2} dx
    # <x^k> = integral x^k |P(x)|^2 e^{-x^2} dx / ||g||_2^2
    
    # For the uncertainty product, we need:
    # sigma_x^2 = <x^2> - <x>^2
    # sigma_p^2 = <p^2> - <p>^2 where p = -i d/dx
    # For g = P e^{-x^2/2}, g' = (P' - x*P) e^{-x^2/2}
    # p g = -i g' = -i(P' - xP) e^{-x^2/2}
    # |p g|^2 = |P' - xP|^2 e^{-x^2}
    
    # Let's compute the needed integrals using the fact that for even polynomials
    # (all our H_n are even for even n), <x> = 0 and <p> = 0 by symmetry.
    
    # So sigma_x^2 = <x^2> = integral x^2 |P|^2 e^{-x^2} dx / integral |P|^2 e^{-x^2} dx
    # sigma_p^2 = <p^2> = integral |P' - xP|^2 e^{-x^2} dx / integral |P|^2 e^{-x^2} dx
    
    # The product sigma_x * sigma_p is what we want to minimize (upper bound C4).
    
    # To compute these integrals, I'll use Gauss-Hermite quadrature.
    # integral f(x) e^{-x^2} dx ~ sum w_i f(x_i)
    
    # Use high-order Gauss-Hermite quadrature
    n_quad = 200
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    # Evaluate P and P' at quadrature nodes
    x = x_nodes
    P_vals = np.polynomial.polynomial.polyval(x, P[::-1])  # P is highest-first, reverse for polyval
    # Actually, let me use np.polyval which expects highest-first
    P_vals = np.polyval(P, x)
    
    # P'(x)
    P_prime = np.polyder(P)
    if len(P_prime) == 0:
        P_prime_vals = np.zeros_like(x)
    else:
        P_prime_vals = np.polyval(P_prime, x)
    
    # |P|^2 e^{-x^2} and weights
    abs_P_sq = P_vals * P_vals
    integrand_norm = abs_P_sq * w_nodes
    norm_sq = np.sum(integrand_norm)
    
    if norm_sq < 1e-30:
        return 1e10  # Degenerate
    
    # <x^2>
    integrand_x2 = x**2 * abs_P_sq * w_nodes
    x2_moment = np.sum(integrand_x2) / norm_sq
    
    # |P' - xP|^2 e^{-x^2}
    diff = P_prime_vals - x * P_vals
    integrand_p2 = diff * diff * w_nodes
    p2_moment = np.sum(integrand_p2) / norm_sq
    
    sigma_x = np.sqrt(x2_moment)
    sigma_p = np.sqrt(p2_moment)
    
    C4 = sigma_x * sigma_p
    return C4


def construct():
    """Return three coefficients (c0, c1, c2) for H0, H4, H8."""
    deadline = time.time() + 180.0
    
    # Seed point
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_C4(*best_c)
    
    # Try multiple random restarts with L-BFGS-B
    np.random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 30.0:
            break
        
        # Random initial point
        scale = 1.0
        c_init = np.random.randn(3) * scale
        # Ensure c3 won't be too extreme
        H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, -560.0, 7257600.0
        c3_est = -(c_init[0]*H0_0 + c_init[1]*H4_0 + c_init[2]*H8_0) / H12_0
        if abs(c3_est) > 1.0:
            c_init = c_init / (abs(c3_est) + 1)
        
        try:
            result = minimize(
                lambda c: _compute_C4(c[0], c[1], c[2]),
                c_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                lambda c: _compute_C4(c[0], c[1], c[2]),
                best_c,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
