"""Optimized program: numerical minimization of the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial


def _hermite_prob(n, x):
    """Probabilists' Hermite polynomial Hn(x)."""
    # Use recurrence for numerical stability
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _g(x, a0, a4, a8):
    """
    Test function g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    
    g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    H0(0) = 1
    H4(0) = -3
    H8(0) = 105
    H12(0) = -945
    
    So: a0*1 + a4*(-3) + a8*105 + a12*(-945) = 0
    => a12 = (a0 - 3*a4 + 105*a8) / 945
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 945.0
    h0 = _hermite_prob(0, x)
    h4 = _hermite_prob(4, x)
    h8 = _hermite_prob(8, x)
    h12 = _hermite_prob(12, x)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-x ** 2 / 2.0)


def _compute_c4_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(g) * A(ĝ).
    
    A(g) = sup_x |g(x)|
    A(ĝ) = sup_x |ĝ(x)|
    
    For Hermite-Gaussian functions, the Fourier transform is related to
    the function itself via the Hermite polynomial orthogonality.
    
    Specifically, if g(x) = P(x) * exp(-x^2/2), then
    ĝ(ξ) = (2π)^(-1/2) * P(-iξ) * exp(-ξ^2/2)  (up to normalization)
    
    Actually, for the standard Fourier transform F[f](ξ) = ∫ f(x) e^{-2πixξ} dx,
    the Fourier transform of exp(-x^2/2) is (2π)^(-1/2) * exp(-π^2 ξ^2).
    This is messy. Let's use the unitary Fourier transform instead, or 
    work with the known property that Hermite functions are eigenfunctions.
    
    Let's use: F[f](ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx
    
    For g(x) = P(x) e^{-x^2/2}, the Fourier transform can be computed
    by expanding P in Hermite polynomials. Each Hn(x) e^{-x^2/2} has a
    known Fourier transform.
    
    Actually, let's just numerically compute the sup norms.
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Compute g(x) on a grid
    x = np.linspace(-15, 15, 20001)
    h0 = _hermite_prob(0, x)
    h4 = _hermite_prob(4, x)
    h8 = _hermite_prob(8, x)
    h12 = _hermite_prob(12, x)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-x ** 2 / 2.0)
    
    A_g = np.max(np.abs(g))
    
    # Compute Fourier transform of g
    # Using the property: F[P(x) e^{-x^2/2}](ξ) 
    # We can use the fact that the Fourier transform of Hn(x) e^{-x^2/2}
    # is (-i)^n Hn(x) e^{-x^2/2} (with appropriate normalization)
    # 
    # For the unitary Fourier transform F[f](ξ) = 1/√(2π) ∫ f(x) e^{-ixξ} dx:
    # F[Hn(x) e^{-x^2/2}](ξ) = (-i)^n Hn(ξ) e^{-ξ^2/2}
    
    # So if g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x^2/2}
    # then ĝ(ξ) = (a0 (-i)^0 H0(ξ) + a4 (-i)^4 H4(ξ) + a8 (-i)^8 H8(ξ) + a12 (-i)^12 H12(ξ)) e^{-ξ^2/2}
    #            = (a0 H0(ξ) + a4 H4(ξ) + a8 H8(ξ) + a12 H12(ξ)) e^{-ξ^2/2}
    
    # Wait, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1. So actually all coefficients are positive!
    # That means ĝ(ξ) = g(ξ) in this case.
    
    # Let me verify: (-i)^0 = 1, (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1, 
    # (-i)^8 = 1, (-i)^12 = 1. Yes!
    
    # So A_hat_g = A_g, and the bound is A_g^2.
    
    # But wait, this seems too simple. Let me double-check with a direct FFT.
    dx = x[1] - x[0]
    g_fft = np.fft.fftshift(np.fft.fft(g * dx)) / np.sqrt(2 * np.pi)
    # The FFT gives values at frequencies, but the grid is not centered at 0 for FFT.
    # Let's just use the analytical result.
    
    A_hat_g = A_g  # Since all (-i)^n for even n are 1
    
    return A_g * A_hat_g


def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    """
    def objective(coeffs):
        a0, a4, a8 = coeffs
        try:
            return _compute_c4_bound(a0, a4, a8)
        except Exception:
            return 1e10
    
    # Initial guess: the known good solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple starting points to avoid local minima
    best_val = np.inf
    best_x = x0
    
    start_points = [
        x0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.13, 0.04, 0.0003]),
    ]
    
    for sp in start_points:
        try:
            res = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Refine with BFGS if possible
    try:
        res = minimize(
            objective,
            best_x,
            method='BFGS',
            options={'maxiter': 10000, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
