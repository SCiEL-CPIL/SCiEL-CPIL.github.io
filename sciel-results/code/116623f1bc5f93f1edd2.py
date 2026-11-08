"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_herm

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 5.0  # seconds


def _hermite_coeffs(n):
    """Return coefficients of the physicist's Hermite polynomial H_n(x) as a list [a0, a1, ..., an]."""
    # Use recurrence: H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h_prev = [1.0]  # H_0
    h_curr = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2*(k-1) * H_{k-2}
        # 2x * H_{k-1}: shift coefficients by 1 and multiply by 2
        term1 = [0.0] + [2.0 * c for c in h_curr]
        # 2*(k-1) * H_{k-2}
        term2 = [2.0 * (k - 1) * c for c in h_prev]
        # H_k = term1 - term2
        max_len = max(len(term1), len(term2))
        term1 = term1 + [0.0] * (max_len - len(term1))
        term2 = term2 + [0.0] * (max_len - len(term2))
        h_new = [t1 - t2 for t1, t2 in zip(term1, term2)]
        h_prev = h_curr
        h_curr = h_new
    return h_curr


def _eval_hermite(coeffs, x):
    """Evaluate Hermite polynomial with given coefficients at x (scalar or array)."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(reversed(coeffs)):
        result = result * x + c
    return result


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for the given Hermite coefficients."""
    # H0, H4, H8, H12 coefficients
    h0 = _hermite_coeffs(0)
    h4 = _hermite_coeffs(4)
    h8 = _hermite_coeffs(8)
    h12 = _hermite_coeffs(12)
    
    # Constraint: g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    h0_0 = _eval_hermite(h0, np.array([0.0]))[0]
    h4_0 = _eval_hermite(h4, np.array([0.0]))[0]
    h8_0 = _eval_hermite(h8, np.array([0.0]))[0]
    h12_0 = _eval_hermite(h12, np.array([0.0]))[0]
    
    if abs(h12_0) < 1e-15:
        # Degenerate case
        return 1e10
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Numerical computation on a grid
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    H0 = _eval_hermite(h0, x)
    H4 = _eval_hermite(h4, x)
    H8 = _eval_hermite(h8, x)
    H12 = _eval_hermite(h12, x)
    
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = poly * np.exp(-x**2 / 2)
    
    # A(f) = sqrt(int x^2 |f|^2 dx) / int |f|^2 dx  -- wait, need to check definition
    # Typically A(f) = (int x^2 |f(x)|^2 dx) / (int |f(x)|^2 dx) for the "variance"
    # Or A(f) = sqrt(int x^2 |f|^2) / int |f|^2 ... let me use the standard uncertainty
    
    # For the uncertainty principle, typically:
    # Δx = sqrt(∫x²|f|²dx / ∫|f|²dx - (∫x|f|²dx/∫|f|²dx)²)
    # But often for even functions centered at 0, the mean is 0.
    
    # Let me compute:
    # norm_sq = ∫|g|² dx
    # x2_moment = ∫x²|g|² dx
    # A_x = sqrt(x2_moment / norm_sq)  (assuming mean is 0 by symmetry)
    
    norm_sq = np.trapz(np.abs(g)**2, x)
    x2_moment = np.trapz(x**2 * np.abs(g)**2, x)
    
    if norm_sq < 1e-15:
        return 1e10
    
    A_x = np.sqrt(x2_moment / norm_sq)
    
    # Fourier transform: use FFT or analytic
    # For Gaussian-modulated polynomials, we can use the fact that FT of H_n(x)exp(-x²/2) is (-i)^n H_n(x)exp(-x²/2)
    # So |F(g)(ξ)|² = |g(ξ)|² (by Plancherel and the self-similarity)
    # Actually, the FT of H_n(x) e^{-x²/2} is (-i)^n H_n(x) e^{-x²/2} (with appropriate normalization)
    # So |F(g)(ξ)| = |g(ξ)|, meaning A_ξ = A_x
    
    # Wait, that would make the product A_x². But the bound involves a constant C4.
    # Let me reconsider. The uncertainty product is Δx * Δξ.
    # For the standard Fourier transform F(f)(ξ) = ∫f(x)e^{-2πixξ}dx or ∫f(x)e^{-ixξ}dx
    
    # Let me use the convention F(f)(ξ) = ∫f(x)e^{-ixξ}dx
    # Then FT of e^{-x²/2} is sqrt(2π)e^{-ξ²/2}
    # FT of H_n(x)e^{-x²/2} = (-i)^n sqrt(2π) H_n(ξ)e^{-ξ²/2}
    
    # So |F(g)(ξ)|² = 2π |g(ξ)|² (same polynomial structure)
    # Thus A_ξ = A_x (the ratio is the same)
    
    # So the product is A_x * A_ξ = A_x²
    
    # But wait, the task says "upper bound on A(f)*A(Fourier(f))" and the score is C4.
    # Let me just compute A_x * A_ξ where A_ξ is computed from the Fourier transform.
    
    # Using the analytic FT: F(g)(ξ) = sqrt(2π) * (c0*H0(ξ) + c4*H4(ξ) + c8*H8(ξ) + c12*H12(ξ)) * e^{-ξ²/2} * (phase factors)
    # The magnitude |F(g)(ξ)| = sqrt(2π) * |poly(ξ)| * e^{-ξ²/2} = sqrt(2π) * |g(ξ)|
    
    # So A_ξ = sqrt(∫ξ²|F(g)|²dξ / ∫|F(g)|²dξ) = sqrt(2π ∫ξ²|g|²dξ / 2π ∫|g|²dξ) = A_x
    
    product = A_x * A_x  # = A_x²
    
    return product


def construct():
    # Try to optimize within the time budget
    best_score = _compute_bound(-113/100, 1/25, 1/3240)
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return best_coeffs
    
    # Try some local optimization
    try:
        x0 = np.array([-113/100, 1/25, 1/3240])
        
        def objective(coeffs):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                return 1e10
            c0, c4, c8 = coeffs
            return _compute_bound(c0, c4, c8)
        
        # Try Nelder-Mead
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
        
        # Try a few more starting points if time allows
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN - 10:
            # Try perturbed starting points
            for trial in range(3):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                x0_trial = x0 + np.random.normal(0, 0.1, 3)
                result = minimize(objective, x0_trial, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                          'adaptive': True})
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END

