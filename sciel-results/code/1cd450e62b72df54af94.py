"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution
import time


def _compute_bound_fast(a0, a4, a8):
    """
    Compute the uncertainty bound for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    Uses the spectral properties of Hermite-Gaussian functions.
    """
    # H_{2k}(0) = (-1)^k * (2k)! / k!
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = -1680.0
    H12_0 = -103950.0
    
    # g(0) = 0: a0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # For the uncertainty bound, we need to compute:
    # The product of the "spread" in x-space and frequency space
    # 
    # For g(x) = p(x) * exp(-x^2/2), where p(x) is a polynomial:
    # ||g||_2^2 = <g, g> = integral of |g|^2 dx
    # ||x*g||_2^2 = integral of x^2 |g|^2 dx
    # Similarly for the Fourier transform side.
    #
    # The FT of H_n(x)*exp(-x^2/2) with convention F[f](ξ) = ∫ f(x) e^{-ixξ} dx
    # is: (-i)^n * sqrt(2π) * H_n(ξ) * exp(-ξ^2/2) * (1/sqrt(2))^n * (sqrt(2)/sqrt(2π)) 
    # Actually, let me use the known result:
    # F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n sqrt(2π) H_n(ξ) e^{-ξ^2/2} / (2^{n/2} sqrt(2π)) * 2^{n/2}
    # 
    # More carefully: The Hermite functions ψ_n(x) = (2^n n! sqrt(π))^{-1/2} H_n(x) e^{-x^2/2}
    # are eigenfunctions of the Fourier transform with eigenvalue (-i)^n.
    # So F[ψ_n](ξ) = (-i)^n ψ_n(ξ)
    # 
    # Thus F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n sqrt(2^n n! sqrt(π)) H_n(ξ) e^{-ξ^2/2} / sqrt(2^n n! sqrt(π)) * sqrt(2π)
    # = (-i)^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    #
    # Wait, let me be more careful. If ψ_n(x) = c_n H_n(x) e^{-x^2/2} with c_n = (2^n n! sqrt(π))^{-1/2}
    # and F[ψ_n] = (-i)^n ψ_n, then:
    # F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n / c_n * ψ_n(ξ) = (-i)^n / c_n * c_n H_n(ξ) e^{-ξ^2/2}
    # = (-i)^n H_n(ξ) e^{-ξ^2/2}
    #
    # But this is for the unitary FT. For F[f](ξ) = ∫ f(x) e^{-ixξ} dx (non-unitary):
    # F[f](ξ) = sqrt(2π) * F_unitary[f](ξ)
    # So F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    
    # So if g(x) = sum_k a_k H_k(x) e^{-x^2/2}, then:
    # ĝ(ξ) = sqrt(2π) sum_k a_k (-i)^k H_k(ξ) e^{-ξ^2/2}
    
    # Now compute the norms:
    # ||g||_2^2 = sum_{k,l} a_k a_l <H_k e^{-x^2/2}, H_l e^{-x^2/2}>
    # <H_k e^{-x^2/2}, H_l e^{-x^2/2}> = δ_{kl} 2^k k! sqrt(π)
    
    # So ||g||_2^2 = sum_k a_k^2 * 2^k * k! * sqrt(π)
    
    # ||x*g||_2^2 = integral x^2 |g|^2 dx
    # Using x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
    # So x g(x) = sum_k a_k [(1/2) H_{k+1}(x) + k H_{k-1}(x)] e^{-x^2/2}
    # The coefficients for H_{k+1} and H_{k-1} terms...
    
    # Let me just compute numerically for accuracy
    
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    
    # Build the polynomial part
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x**2 / 2.0)
    
    g2 = g**2
    norm_g2 = np.trapz(g2, x)
    
    x2_g2 = x**2 * g2
    x2_norm = np.trapz(x2_g2, x)
    
    # Fourier side: ĝ(ξ) = sqrt(2π) * [a0*H0(ξ) + a4*(-i)^4*H4(ξ) + a8*(-i)^8*H8(ξ) + a12*(-i)^12*H12(ξ)] * e^{-ξ^2/2}
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^8 * (-i)^4 = 1
    # So all coefficients are real and positive!
    # ĝ(ξ) = sqrt(2π) * [a0*H0(ξ) + a4*H4(ξ) + a8*H8(ξ) + a12*H12(ξ)] * e^{-ξ^2/2}
    
    # Wait, that's the same polynomial! So ĝ(ξ) = sqrt(2π) * g(ξ) (with the same coefficients)
    # That means the Fourier transform has the same form!
    
    # So ||ĝ||_2^2 = 2π * ||g||_2^2
    # And ||ξ*ĝ||_2^2 = 2π * ||x*g||_2^2
    
    # The uncertainty product:
    # σ_x * σ_ξ = sqrt(x2_norm/norm_g2) * sqrt(xi2_norm/norm_g2_hat)
    # = sqrt(x2_norm/norm_g2) * sqrt(2π * x2_norm / (2π * norm_g2))
    # = sqrt(x2_norm/norm_g2) * sqrt(x2_norm/norm_g2)
    # = x2_norm / norm_g2
    
    # So the uncertainty product is simply x2_norm / norm_g2 = <x^2> (the variance-like quantity)
    
    if norm_g2 < 1e-15:
        return 1e10
    
    C = x2_norm / norm_g2
    
    return C


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the uncertainty bound.
    """
    start_time = time.time()
    deadline = start_time + 180
    
    # Start from the known good point
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_x = x0.copy()
    best_val = _compute_bound_fast(*x0)
    
    # Try structured starting points
    candidates = [
        x0,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.5, 0.1]),
        np.array([0.5, -0.3, 0.2]),
        np.array([-0.5, 0.2, -0.1]),
        np.array([1.0, -0.5, 0.3]),
        np.array([-0.2, 0.1, -0.01]),
        np.array([0.1, -0.05, 0.005]),
    ]
    
    for start in candidates:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_compute_bound_fast, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Random restarts
    np.random.seed(42)
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > deadline - 30:
            break
        try:
            rand_start = np.random.randn(3) * 3.0
            res = minimize(_compute_bound_fast, rand_start, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_compute_bound_fast, best_x, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return np.array(best_x)

# EVOLVE-BLOCK-END
