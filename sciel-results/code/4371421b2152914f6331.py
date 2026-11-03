"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute the physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2 = h_nm1
        h_nm1 = h_n
    return h_nm1


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty product bound is related to the variances in position and momentum.
    For a function g, we compute:
      A(g) = sqrt( <x^2> )  (position spread)
      A(G) = sqrt( <p^2> )  (momentum spread, where G is Fourier transform)
    
    For Fourier transform convention: G(k) = (1/sqrt(2*pi)) * integral g(x) e^{-ikx} dx
    Then <p^2> = <(-i d/dx)^2> = -<d^2/dx^2> in the appropriate normalization.
    
    Actually, let's think more carefully. The uncertainty principle states:
      Delta(x) * Delta(p) >= 1/2
    
    But the task says "upper bound on A(f)*A(Fourier(f))". Let me interpret this as:
    We have a test function g, and we want to minimize the product of its spread
    in position space and its spread in frequency space.
    
    For g(x) = P(x) * exp(-x^2/2), where P is a polynomial:
      <x^2> = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx
      <k^2> = integral k^2 |G(k)|^2 dk / integral |G(k)|^2 dk
    
    By Parseval, integral |G(k)|^2 dk = integral |g(x)|^2 dx.
    
    Also, <k^2> in frequency space = <(-i d/dx)^2> in position space = -<d^2/dx^2>
    So <k^2> = - integral g*(x) g''(x) dx / integral |g(x)|^2 dx
    = integral |g'(x)|^2 dx / integral |g(x)|^2 dx  (by integration by parts, assuming boundary terms vanish)
    
    So we need to compute:
      norm = integral |g|^2 dx
      x2 = integral x^2 |g|^2 dx
      gprime2 = integral |g'|^2 dx
    
    Then A(g)*A(G) = sqrt(x2/norm) * sqrt(gprime2/norm)
    
    The constraint is g(0) = 0, which determines c12 in terms of c0, c4, c8.
    
    Let's use Gauss-Hermite quadrature for efficient computation.
    """
    # Choose c12 so that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1
    # H4(0) = -12  (H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12)
    # Wait, let me compute: H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12
    # H8(0): H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680, so H8(0) = 1680
    # H12(0): H12(x) = ... H12(0) = 12! * (coefficient of x^0)... 
    # Let me compute using recurrence at x=0:
    # H0(0) = 1
    # H1(0) = 0
    # H2(0) = 2*0*H1(0) - 2*1*H0(0) = -2
    # H3(0) = 2*0*H2(0) - 2*2*H1(0) = 0
    # H4(0) = 2*0*H3(0) - 2*3*H2(0) = -6*(-2) = 12
    # H5(0) = 2*0*H4(0) - 2*4*H3(0) = 0
    # H6(0) = 2*0*H5(0) - 2*5*H4(0) = -10*12 = -120
    # H7(0) = 0
    # H8(0) = 2*0*H7(0) - 2*7*H6(0) = -14*(-120) = 1680
    # H9(0) = 0
    # H10(0) = 2*0*H9(0) - 2*9*H8(0) = -18*1680 = -30240
    # H11(0) = 0
    # H12(0) = 2*0*H11(0) - 2*11*H10(0) = -22*(-30240) = 665280
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # g(0) = c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Use Gauss-Hermite quadrature with enough nodes
    # For polynomials up to degree 12, we need at least 7 nodes for exact integration
    # of |g|^2 which is degree 24, so we need 13 nodes. Let's use 50 for safety.
    n_nodes = 50
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    
    # Compute H0, H4, H8, H12 at quadrature points
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 2)
    
    # |g|^2 = P^2 * exp(-x^2)
    g2 = g**2
    
    # norm = integral |g|^2 dx
    # Gauss-Hermite: integral f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
    # Here we have exp(-x^2) already, so we need to be careful.
    # Actually, Gauss-Hermite quadrature: integral_{-inf}^{inf} f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
    # Our integrand is |g(x)|^2 = P(x)^2 * exp(-x^2), so f(x) = P(x)^2
    norm = np.sum(w * P**2)
    
    if norm < 1e-15:
        return 1e15  # Invalid, return large value
    
    # x2 = integral x^2 |g|^2 dx = integral x^2 P(x)^2 exp(-x^2) dx
    x2 = np.sum(w * x**2 * P**2)
    
    # g'(x) = P'(x) * exp(-x^2/2) + P(x) * (-x) * exp(-x^2/2)
    #       = (P'(x) - x*P(x)) * exp(-x^2/2)
    # |g'|^2 = (P'(x) - x*P(x))^2 * exp(-x^2)
    # P'(x) = c0*H0' + c4*H4' + c8*H8' + c12*H12'
    # H_n'(x) = 2n * H_{n-1}(x)
    H0p = _hermite_poly(0, x) * 0  # H0' = 0
    H4p = 2 * 4 * _hermite_poly(3, x)
    H8p = 2 * 8 * _hermite_poly(7, x)
    H12p = 2 * 12 * _hermite_poly(11, x)
    
    Pp = c0 * H0p + c4 * H4p + c8 * H8p + c12 * H12p
    
    gprime = (Pp - x * P) * np.exp(-x**2 / 2)
    gprime2 = gprime**2
    
    # integral |g'|^2 dx = integral (P' - xP)^2 exp(-x^2) dx
    gprime2_int = np.sum(w * (Pp - x * P)**2)
    
    # A(g) = sqrt(x2/norm), A(G) = sqrt(gprime2_int/norm)
    A_g = np.sqrt(x2 / norm)
    A_G = np.sqrt(gprime2_int / norm)
    
    return A_g * A_G


def construct():
    # Initial guess from the prior state of the art
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    # We can normalize the coefficients since the objective is homogeneous of degree 0
    # (scaling all coefficients by a constant doesn't change the ratio)
    # So we can fix one coefficient, say c0 = 1, and optimize c4/c0 and c8/c0
    # Or we can just optimize the three coefficients with a normalization constraint
    
    # Let's optimize with a normalization constraint: c0^2 + c4^2 + c8^2 = 1
    # Actually, since the objective is scale-invariant, we can just set c0 = 1 and
    # optimize c4 and c8
    
    def objective(params):
        c0 = params[0]
        c4 = params[1]
        c8 = params[2]
        return _compute_bound(c0, c4, c8)
    
    # Start from the known good solution
    x0 = np.array([c0_init, c4_init, c8_init])
    
    # Use Nelder-Mead for derivative-free optimization
    start_time = time.time()
    timeout = 150  # Leave some margin before the 210s limit
    
    best_result = None
    best_func = objective(x0)
    
    # Try Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result.fun < best_func:
            best_func = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Try BFGS with numerical gradient
    try:
        result2 = minimize(objective, x0, method='BFGS',
                          options={'maxiter': 10000, 'gtol': 1e-12})
        if result2.fun < best_func:
            best_func = result2.fun
            best_result = result2.x
    except Exception:
        pass
    
    # Try L-BFGS-B
    try:
        result3 = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 10000, 'gtol': 1e-12})
        if result3.fun < best_func:
            best_func = result3.fun
            best_result = result3.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout:
        if best_result is not None:
            return np.array(best_result)
        else:
            return x0
    
    # Try a few random restarts if we have time
    np.random.seed(42)
    for i in range(20):
        if time.time() - start_time > timeout:
            break
        # Random perturbation
        scale = 0.1 * (i + 1)
        x_rand = x0 + scale * np.random.randn(3)
        try:
            result_rand = minimize(objective, x_rand, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result_rand.fun < best_func:
                best_func = result_rand.fun
                best_result = result_rand.x
        except Exception:
            pass
    
    if best_result is not None:
        return np.array(best_result)
    else:
        return x0
# EVOLVE-BLOCK-END
