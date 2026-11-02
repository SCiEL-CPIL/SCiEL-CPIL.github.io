"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_coefficients(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x).
    H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, etc.
    Returns list of coefficients [a0, a1, ..., an] for sum ai * x^i."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    # Use recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
    coeffs_prev = [1.0]  # H_0
    coeffs_curr = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2*(k-1) * H_{k-2}
        new_coeffs = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift coeffs_curr by 1 and multiply by 2
        for i in range(len(coeffs_curr)):
            new_coeffs[i + 1] += 2.0 * coeffs_curr[i]
        # -2*(k-1) * H_{k-2}
        for i in range(len(coeffs_prev)):
            new_coeffs[i] -= 2.0 * (k - 1) * coeffs_prev[i]
        coeffs_prev = coeffs_curr
        coeffs_curr = new_coeffs
    return coeffs_curr


def _poly_eval(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _compute_upper_bound(c0, c4, c8):
    """Compute the upper bound on C4 given Hermite coefficients c0, c4, c8.
    
    The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 chosen so that g(0) = 0.
    
    The upper bound involves integrals of g and its Fourier transform.
    For Hermite-Gaussian functions, the Fourier transform properties simplify.
    
    A(f) = sup_x |f(x)| / ||f||_2 (or similar norm ratio)
    The uncertainty principle bound is A(f) * A(F) >= C4
    
    For the specific structure of this problem, the bound C4 can be computed
    from the coefficients. Based on the literature on uncertainty inequalities
    with Hermite polynomials, the bound involves the L-infinity norms and
    L2 norms of the test function and its Fourier transform.
    
    Let's compute this properly. The key quantities are:
    - ||g||_inf / ||g||_2 (for A(g))
    - ||F(g)||_inf / ||F(g)||_2 (for A(F(g)))
    
    Since F is unitary on L2, ||F(g)||_2 = ||g||_2.
    
    For g(x) = P(x) * exp(-x^2/2) where P is a polynomial:
    ||g||_2^2 = integral |P(x)|^2 exp(-x^2) dx
    
    The Fourier transform of x^n * exp(-x^2/2) can be computed.
    
    Actually, let me use the fact that for the specific Hermite-Gaussian basis:
    The L2 norm of H_n(x) * exp(-x^2/2) is sqrt(pi * n! * 2^n)
    
    Wait, let me be more careful. The standard normalization:
    h_n(x) = H_n(x) * exp(-x^2/2)
    ||h_n||_2^2 = integral H_n(x)^2 exp(-x^2) dx = sqrt(pi) * 2^n * n!
    
    For the Fourier transform: F[H_n(x) exp(-x^2/2)] = (-i)^n H_n(x) exp(-x^2/2)
    (with the convention F[f](xi) = integral f(x) exp(-2*pi*i*x*xi) dx, 
     but for the symmetric convention it's different)
    
    Let me use the convention where F[f](xi) = (1/sqrt(2*pi)) integral f(x) exp(-i*x*xi) dx
    Then F[exp(-x^2/2)] = exp(-xi^2/2)
    And F[H_n(x) exp(-x^2/2)] = (-i)^n H_n(xi) exp(-xi^2/2)
    
    So ||F(g)||_2 = ||g||_2 for any combination.
    
    For the L-infinity norm, we need to find sup_x |g(x)|.
    
    This is a non-trivial optimization. Let me set up the problem numerically.
    """
    # Get Hermite polynomial coefficients
    h0 = _hermite_coefficients(0)
    h4 = _hermite_coefficients(4)
    h8 = _hermite_coefficients(8)
    h12 = _hermite_coefficients(12)
    
    # c12 is determined by g(0) = 0:
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0) = 0
    # H0(0) = 1, H4(0) = -12, H8(0) = 1008, H12(0) = -10395
    h0_0 = _poly_eval(h0, 0.0)
    h4_0 = _poly_eval(h4, 0.0)
    h8_0 = _poly_eval(h8, 0.0)
    h12_0 = _poly_eval(h12, 0.0)
    
    if abs(h12_0) < 1e-15:
        return 1e10  # Degenerate case
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Now compute the bound. We need:
    # A(g) = sup_x |g(x)| / ||g||_2
    # A(F(g)) = sup_x |F(g)(x)| / ||F(g)||_2 = sup_x |F(g)(x)| / ||g||_2
    # C4 = A(g) * A(F(g)) = (sup |g| * sup |F(g)|) / ||g||_2^2
    
    # Compute ||g||_2^2 = integral |P(x)|^2 exp(-x^2) dx
    # where P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    # We can compute this using Gaussian quadrature or by expanding.
    # Let's use numerical integration with Gauss-Hermite quadrature.
    
    # For the L-infinity norms, we need to find the maxima of |g(x)| and |F(g)(x)|.
    # g(x) = P(x) * exp(-x^2/2)
    # F(g)(x) = Q(x) * exp(-x^2/2) where Q involves the Fourier transform
    
    # Actually, for the specific Fourier convention used in uncertainty principles:
    # If F[f](xi) = integral f(x) exp(-2*pi*i*x*xi) dx, then:
    # F[exp(-pi*x^2)] = exp(-pi*xi^2)
    
    # Let me use a different approach. The bound C4 for the uncertainty inequality
    # A(f)*A(F) >= C4 where A(f) = ||f||_inf / ||f||_2 is related to the
    # specific test function.
    
    # Let me compute numerically using a grid and interpolation.
    
    # First, compute ||g||_2^2 using Gauss-Hermite quadrature
    # integral P(x)^2 * exp(-x^2) dx
    # Using Gauss-Hermite: integral f(x) exp(-x^2) dx ~ sum w_i f(x_i)
    
    x_hermite, w_hermite = np.polynomial.hermite.hermgauss(100)
    
    # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    def P(x):
        return (c0 * _poly_eval(h0, x) + c4 * _poly_eval(h4, x) + 
                c8 * _poly_eval(h8, x) + c12 * _poly_eval(h12, x))
    
    # ||g||_2^2 = integral P(x)^2 * exp(-x^2) dx
    p_vals = P(x_hermite)
    l2_sq = np.sum(w_hermite * p_vals**2)
    
    if l2_sq < 1e-15:
        return 1e10
    
    l2_norm = np.sqrt(l2_sq)
    
    # Now find sup |g(x)| = sup |P(x) * exp(-x^2/2)|
    # and sup |F(g)(x)|
    
    # For the Fourier transform, using the convention F[f](xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*x*xi) dx:
    # F[exp(-x^2/2)] = exp(-xi^2/2)
    # F[x^n * exp(-x^2/2)] can be derived from derivatives
    
    # Actually, let me use the property that for the specific case:
    # The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(xi)*exp(-xi^2/2)
    # (with the appropriate normalization)
    
    # So F(g)(xi) = (c0*(-i)^0*H0(xi) + c4*(-i)^4*H4(xi) + c8*(-i)^8*H8(xi) + c12*(-i)^12*H12(xi)) * exp(-xi^2/2)
    # = (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi)) * exp(-xi^2/2)
    # since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    
    # Wait, (-i)^0 = 1, (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1, 
    # (-i)^8 = ((-i)^4)^2 = 1, (-i)^12 = ((-i)^4)^3 = 1
    # So F(g)(xi) = P(xi) * exp(-xi^2/2) = g(xi)
    
    # That means g is its own Fourier transform (up to sign for odd powers, but we only have even)
    # So sup |F(g)| = sup |g|
    
    # Therefore C4 = (sup |g|)^2 / ||g||_2^2
    
    # Find sup |g(x)| = sup |P(x) * exp(-x^2/2)|
    # Use a fine grid and local optimization
    
    x_grid = np.linspace(-15, 15, 10000)
    g_vals = P(x_grid) * np.exp(-x_grid**2 / 2)
    max_abs_g = np.max(np.abs(g_vals))
    
    # Refine the maximum using local optimization
    # Find the point with maximum |g|
    idx_max = np.argmax(np.abs(g_vals))
    x_start = x_grid[idx_max]
    
    def neg_abs_g(x):
        return -abs(P(np.array(x)) * np.exp(-x[0]**2 / 2))
    
    result = minimize(neg_abs_g, [x_start], method='Nelder-Mead', 
                      options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
    
    if result.success:
        max_abs_g = -result.fun
    else:
        # Keep the grid estimate
        pass
    
    # C4 = (max_abs_g)^2 / l2_sq
    c4_bound = (max_abs_g ** 2) / l2_sq
    
    return c4_bound


def construct():
    """Optimize the three free Hermite coefficients to minimize the upper bound on C4."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Compute initial bound
    best_bound = _compute_upper_bound(*x0)
    best_x = x0.copy()
    
    # Objective function
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            return best_bound  # Return current best to signal we're near deadline
        return _compute_upper_bound(x[0], x[1], x[2])
    
    # Try multiple optimization methods and starting points
    start_points = [
        x0,
        np.array([-1.0, 0.05, 0.001]),
        np.array([-0.5, 0.02, 0.0005]),
        np.array([-2.0, 0.1, 0.002]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.1, 0.04, 0.0012]),
    ]
    
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    
    for sp in start_points:
        for method in methods:
            elapsed = time.time() - start_time
            if elapsed > deadline - 15.0:
                break
            
            try:
                if method == 'Nelder-Mead':
                    result = minimize(objective, sp, method=method,
                                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500})
                elif method == 'Powell':
                    result = minimize(objective, sp, method=method,
                                      options={'xtol': 1e-8, 'ftol': 1e-10, 'maxiter': 500})
                else:  # COBYLA
                    result = minimize(objective, sp, method=method,
                                      options={'maxiter': 500, 'rhobeg': 0.1})
                
                if result.success and result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
                elif not result.success and result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
    
    # Final polish with tighter tolerances if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        try:
            result = minimize(objective, best_x, method='Nelder-Mead',
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000})
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
