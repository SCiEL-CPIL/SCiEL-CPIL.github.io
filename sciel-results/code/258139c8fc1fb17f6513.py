"""Combined program: analytic Hermite integrals + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 170.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12


def _poly_coeffs(c0, c4, c8):
    """
    Return the coefficients of p(x) = c0*He0 + c4*He4 + c8*He8 + c12*He12
    as a numpy array [a12, a10, a8, a6, a4, a2, a0] (even polynomial).
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    a12 = c12
    a10 = -42.0 * c12
    a8 = 1260.0 * c12 + c8
    a6 = -16800.0 * c12 - 28.0 * c8
    a4 = 113400.0 * c12 + 210.0 * c8 + c4
    a2 = -302400.0 * c12 - 420.0 * c8 - 6.0 * c4
    a0 = 10395.0 * c12 + 105.0 * c8 + 3.0 * c4 + c0
    return np.array([a12, a10, a8, a6, a4, a2, a0])


def _find_real_roots(coeffs):
    """Find all real roots of a polynomial given by coeffs (highest degree first)."""
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    return np.sort(real_roots)


def _compute_l1_norm(c0, c4, c8):
    """
    Compute A(f) = ∫|p(x) * exp(-x^2/2)| dx using piecewise Gauss-Legendre quadrature.
    """
    coeffs = _poly_coeffs(c0, c4, c8)
    roots = _find_real_roots(coeffs)
    
    x_min, x_max = -15.0, 15.0
    
    breakpoints = [x_min]
    for r in roots:
        if x_min < r < x_max:
            breakpoints.append(r)
    breakpoints.append(x_max)
    
    n_quad = 64
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        if b - a < 1e-14:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x_nodes = mid + half * xg
        w_nodes = half * wg
        
        x2 = x_nodes * x_nodes
        p_val = (coeffs[0] * x2**6 + coeffs[1] * x2**5 + coeffs[2] * x2**4 +
                 coeffs[3] * x2**3 + coeffs[4] * x2**2 + coeffs[5] * x2 + coeffs[6])
        
        f_val = p_val * np.exp(-x2 / 2.0)
        
        total += np.sum(w_nodes * np.abs(f_val))
    
    return total


def _compute_bound_fast(c0, c4, c8):
    """
    Compute the upper bound A(f)*A(F(f)).
    """
    a_f = _compute_l1_norm(c0, c4, c8)
    return a_f * a_f


def _compute_bound_verify(c0, c4, c8):
    """
    Verification using FFT on a fine grid (slower, for final check).
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    x = np.linspace(-15, 15, 100001)
    dx = x[1] - x[0]
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 42*x2**5 + 1260*x2**4 - 16800*x2**3 + 113400*x2**2 - 302400*x2 + 10395
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = p * np.exp(-x2 / 2.0)
    A_f = np.sum(np.abs(f)) * dx
    
    N = len(x)
    F_f = np.fft.fft(f) * dx
    F_f = np.fft.fftshift(F_f)
    freqs = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    freqs = np.fft.fftshift(freqs)
    dxi = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    A_F = np.sum(np.abs(F_f)) * dxi
    
    return A_f * A_F


def _smooth_abs_grad(params, coeffs, xg, wg, breakpoints, sigma=0.01):
    """
    Compute smooth approximation of A(f)^2 and its gradient.
    Uses soft-absolute: |z| ≈ sqrt(z^2 + sigma^2)
    """
    c0, c4, c8 = params
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Precompute Hermite values at quadrature nodes
    x2 = xg * xg
    H0 = np.ones_like(x2)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 42*x2**5 + 1260*x2**4 - 16800*x2**3 + 113400*x2**2 - 302400*x2 + 10395
    
    # p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + (-3)*c4 + 105*c8) / (-10395) = (c0 - 3*c4 + 105*c8) / 10395
    # So p(x) = c0*(H0 + H12/10395) + c4*(H4 - 3*H12/10395) + c8*(H8 + 105*H12/10395)
    
    h0_eff = H0 + H12 / 10395.0
    h4_eff = H4 - 3.0 * H12 / 10395.0
    h8_eff = H8 + 105.0 * H12 / 10395.0
    
    p_val = c0 * h0_eff + c4 * h4_eff + c8 * h8_eff
    f_val = p_val * np.exp(-x2 / 2.0)
    
    # Soft absolute: sqrt(f^2 + sigma^2)
    soft_abs = np.sqrt(f_val**2 + sigma**2)
    
    # Integrate piecewise
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        if b - a < 1e-14:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x_nodes = mid + half * xg
        w_nodes = half * wg
        
        x2_n = x_nodes * x_nodes
        h0_n = np.ones_like(x2_n)
        h4_n = x2_n**2 - 6*x2_n + 3
        h8_n = x2_n**4 - 28*x2_n**3 + 210*x2_n**2 - 420*x2_n + 105
        h12_n = x2_n**6 - 42*x2_n**5 + 1260*x2_n**4 - 16800*x2_n**3 + 113400*x2_n**2 - 302400*x2_n + 10395
        
        h0_eff_n = h0_n + h12_n / 10395.0
        h4_eff_n = h4_n - 3.0 * h12_n / 10395.0
        h8_eff_n = h8_n + 105.0 * h12_n / 10395.0
        
        p_n = c0 * h0_eff_n + c4 * h4_eff_n + c8 * h8_eff_n
        f_n = p_n * np.exp(-x2_n / 2.0)
        soft_abs_n = np.sqrt(f_n**2 + sigma**2)
        
        total += np.sum(w_nodes * soft_abs_n)
    
    # Gradient: d/dparams [total^2] = 2 * total * d(total)/dparams
    # d(total)/dparams = sum over intervals of sum w_nodes * d(soft_abs)/dparams
    # d(soft_abs)/dparams = f_n * (h0_eff_n, h4_eff_n, h8_eff_n) * exp(-x2/2) / soft_abs_n
    
    grad = np.zeros(3)
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        if b - a < 1e-14:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x_nodes = mid + half * xg
        w_nodes = half * wg
        
        x2_n = x_nodes * x_nodes
        h0_n = np.ones_like(x2_n)
        h4_n = x2_n**2 - 6*x2_n + 3
        h8_n = x2_n**4 - 28*x2_n**3 + 210*x2_n**2 - 420*x2_n + 105
        h12_n = x2_n**6 - 42*x2_n**5 + 1260*x2_n**4 - 16800*x2_n**3 + 113400*x2_n**2 - 302400*x2_n + 10395
        
        h0_eff_n = h0_n + h12_n / 10395.0
        h4_eff_n = h4_n - 3.0 * h12_n / 10395.0
        h8_eff_n = h8_n + 105.0 * h12_n / 10395.0
        
        p_n = c0 * h0_eff_n + c4 * h4_eff_n + c8 * h8_eff_n
        f_n = p_n * np.exp(-x2_n / 2.0)
        soft_abs_n = np.sqrt(f_n**2 + sigma**2)
        
        # d(soft_abs)/dc_j = f_n * h_j_eff_n * exp(-x2/2) / soft_abs_n
        # But f_n already includes exp(-x2/2), so:
        # d(soft_abs)/dc_j = p_n * h_j_eff_n * exp(-x2/2) / soft_abs_n
        exp_term = np.exp(-x2_n / 2.0)
        
        grad[0] += np.sum(w_nodes * f_n * h0_eff_n * exp_term / soft_abs_n)
        grad[1] += np.sum(w_nodes * f_n * h4_eff_n * exp_term / soft_abs_n)
        grad[2] += np.sum(w_nodes * f_n * h8_eff_n * exp_term / soft_abs_n)
    
    # Chain rule: d(A^2)/dc = 2*A * dA/dc
    grad = 2.0 * total * grad
    
    return total * total, grad


def construct():
    """
    Find the three free coefficients (c0, c4, c8) for the Hermite combination
    to minimize the upper bound on A(f)*A(Fourier(f)).
    """
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed with fast method
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Setup quadrature for gradient-based optimization
    n_quad = 64
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    
    # Build breakpoints from seed's roots
    coeffs_seed = _poly_coeffs(*seed)
    roots_seed = _find_real_roots(coeffs_seed)
    x_min, x_max = -15.0, 15.0
    breakpoints = [x_min]
    for r in roots_seed:
        if x_min < r < x_max:
            breakpoints.append(r)
    breakpoints.append(x_max)
    breakpoints = np.array(breakpoints)
    
    # Gradient-based optimization with smooth surrogate
    if _time_left() > 30:
        try:
            sigma = 0.01
            res = minimize(
                _smooth_abs_grad,
                seed,
                args=(coeffs_seed, xg, wg, breakpoints, sigma),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Refine with tighter sigma
    if _time_left() > 20:
        try:
            sigma = 0.001
            res = minimize(
                _smooth_abs_grad,
                best_params,
                args=(coeffs_seed, xg, wg, breakpoints, sigma),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on the exact (non-smooth) objective
    if _time_left() > 15:
        try:
            res_final = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
