"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite


def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x) as array [a0, a1, ..., an]."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    h_prev2 = np.array([1.0])
    h_prev1 = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        h_curr = np.zeros(k + 1)
        h_curr[1:] += 2.0 * h_prev1[:k]
        h_curr[:k-1] -= 2.0 * (k - 1) * h_prev2[:k-1]
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


def _poly_eval(coeffs, x):
    """Evaluate polynomial with coeffs [a0, a1, ..., an] at points x."""
    result = np.zeros_like(x, dtype=float)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute C4 = A(f) * A(F) analytically using Hermite function expansion.
    
    g(x) = P(x) * exp(-x^2/2) where P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    
    We expand g in the orthonormal Hermite function basis:
    h_n(x) = (2^n n! sqrt(pi))^(-1/2) * H_n(x) * exp(-x^2/2)
    
    Then g(x) = sum_n a_n * h_n(x) where a_n = <g, h_n>
    
    Since g = P * exp(-x^2/2), and P is a polynomial of degree 12,
    g only has components in h_0, h_4, h_8, h_12 (by orthogonality).
    
    a_n = integral g(x) * h_n(x) dx = integral P(x) * exp(-x^2/2) * h_n(x) dx
        = integral P(x) * (2^n n! sqrt(pi))^(-1/2) * H_n(x) * exp(-x^2) dx
    
    For the Fourier side, the Fourier transform of h_n is (-i)^n * h_n (for physicist's convention).
    So |G(k)|^2 has the same coefficient structure.
    
    A(f) = sum_n |a_n|^2 * <x>_{h_n} / sum_n |a_n|^2
    where <x>_{h_n} = integral |x| * h_n(x)^2 dx
    
    Similarly for A(F) with the Fourier-side coefficients.
    """
    # Get Hermite polynomial coefficients
    h0_coeffs = _hermite_coeffs(0)
    h4_coeffs = _hermite_coeffs(4)
    h8_coeffs = _hermite_coeffs(8)
    h12_coeffs = _hermite_coeffs(12)
    
    # Evaluate at 0 to get c12
    h0_0 = _poly_eval(h0_coeffs, np.array([0.0]))[0]
    h4_0 = _poly_eval(h4_coeffs, np.array([0.0]))[0]
    h8_0 = _poly_eval(h8_coeffs, np.array([0.0]))[0]
    h12_0 = _poly_eval(h12_coeffs, np.array([0.0]))[0]
    
    if abs(h12_0) < 1e-15:
        return 1e10
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Coefficients of P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P(x) = sum_j p_j * x^j
    max_deg = 12
    p_coeffs = np.zeros(max_deg + 1)
    p_coeffs += c0 * h0_coeffs
    p_coeffs += c4 * h4_coeffs
    p_coeffs += c8 * h8_coeffs
    p_coeffs += c12 * h12_coeffs
    
    # Now compute a_n = <g, h_n> for n = 0, 4, 8, 12
    # a_n = integral P(x) * exp(-x^2/2) * h_n(x) dx
    #     = (2^n n! sqrt(pi))^(-1/2) * integral P(x) * H_n(x) * exp(-x^2) dx
    #
    # integral P(x) * H_n(x) * exp(-x^2) dx
    # P(x) = sum_j p_j x^j
    # = sum_j p_j * integral x^j * H_n(x) * exp(-x^2) dx
    #
    # We can compute integral x^j * H_n(x) * exp(-x^2) dx using recurrence relations
    # or direct numerical quadrature for the small number of terms needed.
    
    # For efficiency, let's use a different approach:
    # Since g(x) = P(x)*exp(-x^2/2), and we know the Hermite function expansion,
    # let's compute a_n directly via numerical integration on a fine grid,
    # but only for the 4 relevant n values. This is fast enough.
    
    # Actually, let's use the fact that we only need 4 coefficients and compute them
    # via Gaussian quadrature with the appropriate weight.
    
    # integral f(x) * exp(-x^2) dx can be computed via Gauss-Hermite quadrature
    # with weight exp(-x^2). scipy has this.
    
    from scipy.integrate import quad
    
    def _int_xj_Hn_expj2(j, n, x):
        """Compute integral x^j * H_n(x) * exp(-x^2) dx from -inf to inf."""
        # Use Gauss-Hermite quadrature: integral f(x) exp(-x^2) dx = sum w_i f(x_i)
        # We need integral x^j * H_n(x) * exp(-x^2) dx
        # = sum_i w_i * x_i^j * H_n(x_i)
        # But we need enough points for degree j+n polynomial
        # Use 50 points which is more than enough for degree up to ~12+12=24
        x_gq, w_gq = np.polynomial.hermite.hermgauss(50)
        Hn_vals = eval_hermite(n, x_gq)
        return np.sum(w_gq * x_gq**j * Hn_vals)
    
    # Precompute the needed integrals
    # a_n = (2^n n! sqrt(pi))^(-1/2) * sum_j p_j * integral x^j * H_n(x) * exp(-x^2) dx
    
    from math import factorial, sqrt, pi
    
    n_values = [0, 4, 8, 12]
    a_coeffs = []
    
    for n in n_values:
        norm_factor = sqrt(2.0**n * factorial(n) * pi)
        integral_val = 0.0
        for j in range(max_deg + 1):
            if abs(p_coeffs[j]) > 1e-15:
                integral_val += p_coeffs[j] * _int_xj_Hn_expj2(j, n, x_gq)
        a_n = integral_val / norm_factor
        a_coeffs.append(a_n)
    
    a_arr = np.array(a_coeffs)
    norm_sq = np.sum(a_arr**2)
    
    if norm_sq < 1e-15:
        return 1e10
    
    # Now compute A(f) = sum |a_n|^2 * <x>_{h_n} / sum |a_n|^2
    # where <x>_{h_n} = integral |x| * h_n(x)^2 dx
    #
    # h_n(x)^2 = (2^n n! sqrt(pi))^(-1) * H_n(x)^2 * exp(-x^2)
    # integral |x| * h_n(x)^2 dx = (2^n n! sqrt(pi))^(-1) * integral |x| * H_n(x)^2 * exp(-x^2) dx
    #
    # For even n, H_n(x)^2 is even, so |x| * H_n(x)^2 is even
    # integral |x| * H_n(x)^2 * exp(-x^2) dx from -inf to inf
    # = 2 * integral x * H_n(x)^2 * exp(-x^2) dx from 0 to inf
    #
    # We can use: integral x * H_n(x)^2 * exp(-x^2) dx
    # Use Gauss-Hermite again but need to handle |x|
    # For even n, H_n is even, so H_n^2 is even, |x|*H_n^2*exp(-x^2) is even
    # integral = 2 * integral_0^inf x * H_n(x)^2 * exp(-x^2) dx
    #
    # Alternatively, use the known result:
    # For Hermite functions, <x>_{h_n} = sqrt(n/2) for the "mean absolute value"
    # Actually, let me compute it numerically with Gauss-Hermite.
    
    # integral |x| * H_n(x)^2 * exp(-x^2) dx
    # = integral |x| * H_n(x)^2 * exp(-x^2) dx
    # Use Gauss-Hermite: sum w_i * |x_i| * H_n(x_i)^2
    
    def _mean_abs_x_hermite(n):
        """Compute integral |x| * H_n(x)^2 * exp(-x^2) dx / (2^n n! sqrt(pi))
        which equals <x>_{h_n} = integral |x| * h_n(x)^2 dx"""
        x_gq, w_gq = np.polynomial.hermite.hermgauss(80)
        Hn_vals = eval_hermite(n, x_gq)
        integral = np.sum(w_gq * np.abs(x_gq) * Hn_vals**2)
        norm_factor = 2.0**n * factorial(n) * sqrt(pi)
        return integral / norm_factor
    
    mean_abs_x = np.array([_mean_abs_x_hermite(n) for n in n_values])
    
    A_f = np.sum(a_arr**2 * mean_abs_x) / norm_sq
    
    # For the Fourier side:
    # The Fourier transform of h_n(x) is (-i)^n * h_n(x) (with the convention
    # that F[f](k) = integral f(x) exp(-ikx) dx / sqrt(2*pi))
    # So |G(k)|^2 = sum |a_n|^2 * h_n(k)^2 (the phases cancel)
    # And A(F) = sum |a_n|^2 * <k>_{h_n} / sum |a_n|^2
    # where <k>_{h_n} = integral |k| * h_n(k)^2 dk = same as <x>_{h_n}
    
    # Wait, but the Fourier transform changes the variable. Let me be more careful.
    # If G(k) = F[g](k) = integral g(x) exp(-ikx) dx / sqrt(2*pi)
    # and g(x) = sum a_n h_n(x), then G(k) = sum a_n (-i)^n h_n(k)
    # So |G(k)|^2 = sum |a_n|^2 h_n(k)^2 (cross terms vanish by orthogonality)
    # A(F) = integral |k| |G(k)|^2 dk / integral |G(k)|^2 dk
    #       = sum |a_n|^2 integral |k| h_n(k)^2 dk / sum |a_n|^2
    #       = sum |a_n|^2 <x>_{h_n} / sum |a_n|^2  (same integral, just variable renamed)
    
    A_F = A_f  # Same value due to symmetry
    
    C4 = A_f * A_F
    
    return C4


def construct():
    t_start = time.time()
    deadline = t_start + 190.0
    
    # Precompute Gauss-Hermite nodes for speed
    x_gq, w_gq = np.polynomial.hermite.hermgauss(50)
    
    # Cache the integral computations
    from math import factorial, sqrt, pi
    
    # Precompute integral x^j * H_n(x) * exp(-x^2) dx for needed j, n
    # j ranges 0..12, n in {0, 4, 8, 12}
    integral_cache = {}
    for n in [0, 4, 8, 12]:
        Hn_vals = eval_hermite(n, x_gq)
        for j in range(13):
            integral_cache[(j, n)] = np.sum(w_gq * x_gq**j * Hn_vals)
    
    # Precompute mean_abs_x for each n
    x_gq2, w_gq2 = np.polynomial.hermite.hermgauss(80)
    mean_abs_x_cache = {}
    for n in [0, 4, 8, 12]:
        Hn_vals = eval_hermite(n, x_gq2)
        integral = np.sum(w_gq2 * np.abs(x_gq2) * Hn_vals**2)
        norm_factor = 2.0**n * factorial(n) * sqrt(pi)
        mean_abs_x_cache[n] = integral / norm_factor
    
    # Hermite polynomial coefficients
    h0_coeffs = _hermite_coeffs(0)
    h4_coeffs = _hermite_coeffs(4)
    h8_coeffs = _hermite_coeffs(8)
    h12_coeffs = _hermite_coeffs(12)
    
    h0_0 = _poly_eval(h0_coeffs, np.array([0.0]))[0]
    h4_0 = _poly_eval(h4_coeffs, np.array([0.0]))[0]
    h8_0 = _poly_eval(h8_coeffs, np.array([0.0]))[0]
    h12_0 = _poly_eval(h12_coeffs, np.array([0.0]))[0]
    
    if abs(h12_0) < 1e-15:
        return np.array([0.0, 0.0, 0.0])
    
    def _fast_compute_bound(c0, c4, c8):
        """Fast analytic computation using precomputed tables."""
        c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
        
        # Build P(x) coefficients
        p_coeffs = np.zeros(13)
        p_coeffs += c0 * h0_coeffs
        p_coeffs += c4 * h4_coeffs
        p_coeffs += c8 * h8_coeffs
        p_coeffs += c12 * h12_coeffs
        
        # Compute a_n for n in {0, 4, 8, 12}
        a_coeffs = []
        for n in [0, 4, 8, 12]:
            norm_factor = sqrt(2.0**n * factorial(n) * pi)
            integral_val = 0.0
            for j in range(13):
                if abs(p_coeffs[j]) > 1e-15:
                    integral_val += p_coeffs[j] * integral_cache[(j, n)]
            a_n = integral_val / norm_factor
            a_coeffs.append(a_n)
        
        a_arr = np.array(a_coeffs)
        norm_sq = np.sum(a_arr**2)
        
        if norm_sq < 1e-15:
            return 1e10
        
        mean_abs_x = np.array([mean_abs_x_cache[n] for n in [0, 4, 8, 12]])
        
        A_f = np.sum(a_arr**2 * mean_abs_x) / norm_sq
        C4 = A_f * A_f  # A_F = A_f by symmetry
        
        return C4
    
    # Starting point
    best_c = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _fast_compute_bound(best_c[0], best_c[1], best_c[2])
    
    # Nelder-Mead optimization
    def objective(c):
        return _fast_compute_bound(c[0], c[1], c[2])
    
    # Multiple restarts with different initial points
    rng = np.random.default_rng(42)
    
    # First, try a grid of starting points
    for c0 in np.linspace(-3.0, 3.0, 13):
        if time.time() > deadline:
            break
        for c4 in np.linspace(-0.5, 0.5, 11):
            if time.time() > deadline:
                break
            for c8 in np.linspace(-0.1, 0.1, 9):
                if time.time() > deadline:
                    break
                c_trial = np.array([c0, c4, c8])
                val = _fast_compute_bound(c0, c4, c8)
                if val < best_val:
                    best_val = val
                    best_c = c_trial.copy()
    
    # Local optimization from best point
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # More restarts from perturbed best
    for i in range(20):
        if time.time() > deadline - 5:
            break
        c_start = best_c + rng.normal(0, 0.1, size=3)
        try:
            result = minimize(
                objective,
                c_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
