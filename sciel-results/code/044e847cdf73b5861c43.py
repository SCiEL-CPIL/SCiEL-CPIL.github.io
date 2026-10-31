"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
import scipy.optimize as opt

def _hermite_poly(x, n):
    """Compute physicist's Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)

def _g_from_coeffs(c0, c4, c8):
    """
    Given coefficients for H0, H4, H8, compute the H12 coefficient so that g(0) = 0.
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    So c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0)
    """
    H0_0 = _hermite_poly(0, 0)
    H4_0 = _hermite_poly(0, 4)
    H8_0 = _hermite_poly(0, 8)
    H12_0 = _hermite_poly(0, 12)
    if abs(H12_0) < 1e-15:
        return np.array([c0, c4, c8, 0.0])
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    return np.array([c0, c4, c8, c12])

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 = A(f) * A(F(f)) for the test function.
    
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so g(0) = 0.
    
    The uncertainty inequality bound involves:
    A(f) = integral |f(x)|^2 dx (or related norm)
    A(F(f)) = integral |F(f)(xi)|^2 dxi
    
    For Hermite-Gaussian functions, we can use Parseval and properties of Fourier transform
    of Hermite functions.
    
    The key insight: For f(x) = g(x) * exp(-x^2/2), we need to compute
    the product of L2 norms of f and its Fourier transform, normalized appropriately.
    
    Actually, the standard uncertainty product bound for these test functions in the
    Goncalves et al. context uses:
    
    C4 bound = (integral |g(x)|^2 dx) * (integral |F[g*exp(-x^2/2)](xi)|^2 dxi) / (normalization)
    
    But more precisely, the bound on A(f)*A(F(f)) where A is the area (integral of absolute value)
    or L2 norm squared.
    
    Let me think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))".
    In the context of uncertainty principles, this is typically the product of variances or
    the product of L2 norms in position and frequency space.
    
    For the specific construction with Hermite polynomials times Gaussian:
    f(x) = P(x) * exp(-x^2/2) where P is a polynomial.
    
    The Fourier transform of exp(-x^2/2) * H_n(x) is (i)^n * exp(-xi^2/2) * H_n(xi) (up to constants).
    
    So if f(x) = sum_k c_k * H_k(x) * exp(-x^2/2), then
    F(f)(xi) = sum_k c_k * (i)^k * H_k(xi) * exp(-xi^2/2) (with appropriate normalization).
    
    The L2 norm squared of f is: integral |f(x)|^2 dx = sum_k |c_k|^2 * integral H_k(x)^2 * exp(-x^2) dx
    = sum_k |c_k|^2 * sqrt(pi) * 2^k * k! (for physicist's Hermite)
    
    Wait, let me be more careful. The orthogonality relation for physicist's Hermite polynomials:
    integral_{-inf}^{inf} H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_{mn}
    
    So ||f||_2^2 = integral |sum_k c_k H_k(x) exp(-x^2/2)|^2 dx
    = integral (sum_k c_k H_k(x) exp(-x^2/2)) (sum_j c_j H_j(x) exp(-x^2/2)) dx
    = sum_k |c_k|^2 * integral H_k(x)^2 exp(-x^2) dx
    = sum_k |c_k|^2 * sqrt(pi) * 2^k * k!
    
    Similarly, for the Fourier transform, using the unitary Fourier transform:
    F(f)(xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*xi*x) dx
    
    The Fourier transform of exp(-x^2/2) * H_n(x) is (i)^n * exp(-xi^2/2) * H_n(xi) (for the unitary FT).
    
    So F(f)(xi) = sum_k c_k * (i)^k * exp(-xi^2/2) * H_k(xi)
    
    And ||F(f)||_2^2 = sum_k |c_k|^2 * sqrt(pi) * 2^k * k! (same by Parseval/unitarity)
    
    So the product ||f||_2^2 * ||F(f)||_2^2 = (sum_k |c_k|^2 * sqrt(pi) * 2^k * k!)^2
    
    This is always the same for f and F(f) due to Parseval's theorem!
    
    So the "bound" must be something else. Let me reconsider.
    
    Perhaps A(f) is the L1 norm (area), not L2 norm. Or perhaps it's a different quantity.
    
    In the uncertainty principle context, the bound on the product of variances is what's typically minimized.
    Or perhaps A(f) = integral |f(x)| dx (L1 norm) and A(F(f)) = integral |F(f)(xi)| dxi.
    
    Let me reconsider the problem. The task mentions "upper bound on C4" and the prior art gives specific coefficients.
    The score is the upper bound value, lower is better.
    
    Given the structure, I think the objective function to minimize is:
    B(c0, c4, c8) = some function of the coefficients that gives an upper bound.
    
    Let me try to compute this numerically. I'll define the function and its Fourier transform,
    and compute the relevant norms.
    """
    c12 = _g_from_coeffs(c0, c4, c8)[3]
    coeffs = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    # Define the function on a grid
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    # g(x) = sum c_k * H_k(x) * exp(-x^2/2)
    g = np.zeros_like(x)
    for c, n in zip(coeffs, degrees):
        g += c * _hermite_poly(x, n) * np.exp(-x**2 / 2)
    
    # L1 norm of g (or f = g)
    A_f = np.trapz(np.abs(g), x)
    
    # Fourier transform via FFT
    N = len(x)
    G = np.fft.fft(g) * dx
    freq = np.fft.fftfreq(N, d=dx)
    
    # Need to handle the full frequency range properly
    # The FFT gives frequencies from -pi/dx to pi/dx (approximately)
    # We need integral |F(f)(xi)| dxi over all xi
    
    # Actually, let's use a different approach. Compute the Fourier transform analytically
    # or use a wider grid.
    
    # Let's compute |F(f)| using the property that F[exp(-x^2/2) * H_n(x)] = (i)^n * exp(-xi^2/2) * H_n(xi)
    xi = np.linspace(-10, 10, 20001)
    dxi = xi[1] - xi[0]
    
    F_g = np.zeros_like(xi)
    for c, n in zip(coeffs, degrees):
        F_g += c * (1j)**n * _hermite_poly(xi, n) * np.exp(-xi**2 / 2)
    
    A_F = np.trapz(np.abs(F_g), xi)
    
    # The bound is A(f) * A(F(f))
    return A_f * A_F

def construct():
    # Use optimization to find better coefficients
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds max
    
    # Initial guess from prior art
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Local optimization using Nelder-Mead
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    # Try Nelder-Mead optimization
    try:
        result = opt.minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
        if time.time() < deadline - 10:
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
    except Exception:
        pass
    
    # Try additional local optimization with tighter tolerances if time permits
    if time.time() < deadline - 15:
        try:
            result2 = opt.minimize(objective, best_coeffs, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                            'adaptive': True})
            if result2.fun < best_bound:
                best_bound = result2.fun
                best_coeffs = result2.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
