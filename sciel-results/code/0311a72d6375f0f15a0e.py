"""
Optimize Fourier-invariant Hermite-Gaussian test function coefficients
to minimize the uncertainty product upper bound.

The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is determined by the constraint g(0) = 0.

The uncertainty bound involves A(f)*A(Fourier(f)) which for a normalized
test function relates to ||g||_1^2 / (min|ĝ|^2) or similar ratios.

We use coordinate descent with analytical updates for each coefficient.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import math

# Deadline guard
_DEADLINE = time.time() + 200  # 200s out of 210s budget
_SAFETY_MARGIN = 10.0


def _hermite_polynomials(n_max=12):
    """Compute physicists' Hermite polynomials H_0 through H_12."""
    polys = []
    for n in range(n_max + 1):
        polys.append(eval_hermite(n))
    return polys


def _g_from_coeffs(a0, a4, a8):
    """
    Construct g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    H0(0)=1, H4(0)=0, H8(0)=0, H12(0)=0 for physicists' Hermite polynomials.
    Wait: H_n(0) = 0 for odd n, and H_n(0) = (-1)^(n/2) * n! / (n/2)! for even n.
    H0(0) = 1
    H4(0) = -6
    H8(0) = 105
    H12(0) = -945
    
    So g(0) = (a0*1 + a4*(-6) + a8*105 + a12*(-945)) * 1 = 0
    => a12 = (a0 - 6*a4 + 105*a8) / 945
    """
    a12 = (a0 - 6.0 * a4 + 105.0 * a8) / 945.0
    return a0, a4, a8, a12


def _compute_bound(a0, a4, a8, num_points=2000):
    """
    Compute the uncertainty bound C4 for given coefficients.
    
    The bound on A(f)*A(Fourier(f)) for a test function g is related to:
    ||g||_1 * ||ĝ||_1 / (||g||_2 * ||ĝ||_2) type quantities, or more specifically
    the constant in the uncertainty inequality.
    
    For the Hermite-Gaussian approach, the key quantity is:
    C4 = sup_x |x * g(x)| / inf_x |ĝ(x)|  (or similar)
    
    Actually, the standard formulation: if f = g * h where h is the window,
    then A(f) <= ||g||_1 and A(F(f)) <= ||ĝ||_1, so A(f)*A(F(f)) <= ||g||_1 * ||ĝ||_1.
    
    But we want to MINIMIZE this product. Let's compute ||g||_1 and ||ĝ||_1.
    
    Actually, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    The bound is ||g||_1 * ||ĝ||_1 (or a normalized version).
    
    Let me compute both L1 norms numerically.
    """
    a12 = (a0 - 6.0 * a4 + 105.0 * a8) / 945.0
    
    # Define g(x)
    def g(x):
        return (a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x) +
                a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)) * np.exp(-x**2 / 2.0)
    
    # Compute ||g||_1 = integral |g(x)| dx over R
    # Use adaptive quadrature over a finite range
    try:
        norm_g, _ = quad(lambda x: abs(g(x)), -15, 15, limit=200)
    except Exception:
        return 1e10
    
    if norm_g < 1e-12:
        return 1e10
    
    # Compute ĝ(ξ) = Fourier transform of g
    # Using the convention: ĝ(ξ) = ∫ g(x) e^{-2πiξx} dx
    # For Hermite-Gaussian functions, the FT has a nice form.
    # H_n(x) e^{-x^2/2} has FT related to H_n(2πξ) e^{-(2πξ)^2}
    
    # Actually, let's use the known FT of Hermite functions.
    # The FT of H_n(x) e^{-x^2/2} is (-i)^n H_n(2πξ) e^{-(2πξ)^2} (up to normalization)
    
    # More precisely, with the convention F[f](ξ) = ∫ f(x) e^{-2πiξx} dx:
    # F[H_n(x) e^{-x^2/2}](ξ) = (-2πi)^n H_n(2πξ) e^{-(2πξ)^2} * (1/sqrt(2π)) * ... 
    # This gets complicated. Let me just compute numerically.
    
    # For numerical FT, sample g on a grid and use FFT, or integrate directly.
    # Since g is smooth and decays, we can integrate |ĝ(ξ)| over ξ.
    
    # ĝ(ξ) = ∫ g(x) e^{-2πiξx} dx
    # |ĝ(ξ)| = |∫ g(x) cos(2πξx) dx| + i|∫ g(x) sin(2πξx) dx|
    # Since g is real and even (Hermite polys of even degree times Gaussian),
    # g is even, so ĝ(ξ) is real and even: ĝ(ξ) = ∫ g(x) cos(2πξx) dx
    
    # Compute ||ĝ||_1 = ∫|ĝ(ξ)| dξ
    # By Plancherel or Parseval, ∫|ĝ(ξ)| dξ is not simply related to ||g||_1.
    
    # Let me compute ĝ(ξ) numerically for a range of ξ.
    def g_hat(xi):
        # ∫ g(x) cos(2πxi x) dx (since g is even and real)
        try:
            val, _ = quad(lambda x: g(x) * np.cos(2.0 * np.pi * xi * x), -15, 15, limit=100)
            return val
        except Exception:
            return 0.0
    
    # Integrate |ĝ(ξ)| over ξ
    # Sample at several points and use trapezoidal rule
    xi_vals = np.linspace(-5, 5, 500)
    gh_vals = np.array([g_hat(xi) for xi in xi_vals])
    norm_gh = np.trapz(np.abs(gh_vals), xi_vals)
    
    if norm_gh < 1e-12:
        return 1e10
    
    # The bound is ||g||_1 * ||ĝ||_1
    return norm_g * norm_gh


def _compute_bound_fast(a0, a4, a8):
    """
    Faster computation using the analytic Fourier transform of Hermite-Gaussians.
    
    For the convention F[f](ξ) = ∫ f(x) e^{-2πiξx} dx:
    F[H_n(x) e^{-x^2/2}](ξ) involves H_n evaluated at 2πξ with appropriate scaling.
    
    The key identity: if φ(x) = e^{-x^2/2}, then F[φ](ξ) = sqrt(2π) e^{-(2πξ)^2}
    For Hermite: F[H_n(x) e^{-x^2/2}](ξ) = (-i)^n (2π)^n H_n(2πξ) e^{-(2πξ)^2} / sqrt(2π) * ...
    
    Actually, let me use a cleaner approach. The Hermite functions ψ_n(x) = (2^n n! sqrt(π))^{-1/2} H_n(x) e^{-x^2/2}
    are eigenfunctions of the FT. But our g is not normalized that way.
    
    Let me just use the fact that for even n, the FT of H_n(x)e^{-x^2/2} is real and even,
    and compute it via the integral representation or known formula.
    
    Known: ∫ H_n(x) e^{-x^2/2} e^{-2πiξx} dx
    = ∫ H_n(x) e^{-(x^2/2 + 2πiξx)} dx
    Complete the square: x^2/2 + 2πiξx = (x + 2πiξ)^2/2 + 2π^2ξ^2
    So the integral = e^{-2π^2ξ^2} ∫ H_n(x) e^{-(x+2πiξ)^2/2} dx
    
    This is getting complex. Let me use a different approach: 
    precompute the FT numerically once using FFT on a fine grid, then interpolate.
    """
    a12 = (a0 - 6.0 * a4 + 105.0 * a8) / 945.0
    
    # Use a fixed grid for x and compute g, then FFT for ĝ
    # This is fast and avoids repeated quadrature
    
    # Grid for x
    N = 4096
    x = np.linspace(-12, 12, N)
    dx = x[1] - x[0]
    
    # Compute g(x) on the grid
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    g_vals = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2.0)
    
    # ||g||_1
    norm_g = np.sum(np.abs(g_vals)) * dx
    
    if norm_g < 1e-12:
        return 1e10
    
    # Fourier transform: ĝ(ξ) = ∫ g(x) e^{-2πiξx} dx
    # Using FFT: if we have g sampled at x_j = j*dx, then
    # FFT gives us ĝ at frequencies ξ_k = k/(N*dx)
    # But our x is not centered at 0 in the FFT sense. Let's shift.
    
    # Actually, numpy's FFT assumes x_j = j for j=0..N-1 with spacing 1.
    # For our case: g(x_j) with x_j = x_0 + j*dx
    # ĝ(ξ) ≈ dx * sum_j g(x_j) e^{-2πiξ x_j}
    # = dx * e^{-2πiξ x_0} * sum_j g(x_j) e^{-2πiξ (x_j - x_0)}
    # The sum is the DFT of g_vals at frequency ξ*(x_j - x_0) = ξ * j * dx
    # So ξ_k = k / (N * dx) for the DFT frequencies
    
    fft_vals = np.fft.fft(g_vals) * dx
    # The frequencies are: freq_k = k/(N*dx) for k=0..N/2, and negative for k>N/2
    freqs = np.fft.fftfreq(N, d=dx)
    
    # Shift to center zero frequency
    fft_centered = np.fft.fftshift(fft_vals)
    freqs_centered = np.fft.fftshift(freqs)
    
    # ||ĝ||_1
    norm_gh = np.sum(np.abs(fft_centered)) * (freqs_centered[1] - freqs_centered[0])
    
    if norm_gh < 1e-12:
        return 1e10
    
    return norm_g * norm_gh


def construct():
    """
    Return three coefficients (a0, a4, a8) that minimize the uncertainty bound.
    The degree-12 coefficient is fixed by the constraint g(0) = 0.
    """
    global _DEADLINE
    
    best_score = 1e10
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate the seed
    seed_score = _compute_bound_fast(*best_coeffs)
    best_score = seed_score
    
    # Coordinate descent with analytical/numerical 1D optimization
    # For each coefficient, fix the others and minimize over that one
    
    max_iterations = 50
    tol = 1e-10
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        
        improved = False
        
        for dim in range(3):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break
            
            # Get current value
            current_val = best_coeffs[dim]
            
            # Define 1D objective: fix other coefficients, vary this one
            def obj_1d(val):
                if dim == 0:
                    return _compute_bound_fast(val, best_coeffs[1], best_coeffs[2])
                elif dim == 1:
                    return _compute_bound_fast(best_coeffs[0], val, best_coeffs[2])
                else:
                    return _compute_bound_fast(best_coeffs[0], best_coeffs[1], val)
            
            # Use bounded scalar minimization
            # Choose reasonable bounds
            lower = current_val * 0.5 - 0.1
            upper = current_val * 1.5 + 0.1
            # Ensure symmetric-ish bounds around 0 if current is near 0
            if abs(current_val) < 0.01:
                lower = -1.0
                upper = 1.0
            
            try:
                result = minimize_scalar(obj_1d, bounds=(lower, upper), method='bounded',
                                        options={'xatol': 1e-12, 'maxiter': 100})
                if result.success and result.fun < best_score:
                    best_coeffs[dim] = result.x
                    best_score = result.fun
                    improved = True
            except Exception:
                pass
        
        if not improved:
            # Try a wider search
            for dim in range(3):
                if time.time() > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                current_val = best_coeffs[dim]
                
                def obj_1d_wide(val):
                    if dim == 0:
                        return _compute_bound_fast(val, best_coeffs[1], best_coeffs[2])
                    elif dim == 1:
                        return _compute_bound_fast(best_coeffs[0], val, best_coeffs[2])
                    else:
                        return _compute_bound_fast(best_coeffs[0], best_coeffs[1], val)
                
                # Wider bounds
                for scale in [2.0, 5.0, 10.0]:
                    lower = current_val - scale
                    upper = current_val + scale
                    try:
                        result = minimize_scalar(obj_1d_wide, bounds=(lower, upper), method='bounded',
                                                options={'xatol': 1e-10, 'maxiter': 200})
                        if result.success and result.fun < best_score:
                            best_coeffs[dim] = result.x
                            best_score = result.fun
                            improved = True
                            break
                    except Exception:
                        pass
                
                if not improved:
                    break
            
            if not improved:
                break
    
    # Final polish with tighter tolerance
    if time.time() < _DEADLINE - _SAFETY_MARGIN:
        for dim in range(3):
            current_val = best_coeffs[dim]
            
            def obj_1d_polish(val):
                if dim == 0:
                    return _compute_bound_fast(val, best_coeffs[1], best_coeffs[2])
                elif dim == 1:
                    return _compute_bound_fast(best_coeffs[0], val, best_coeffs[2])
                else:
                    return _compute_bound_fast(best_coeffs[0], best_coeffs[1], val)
            
            lower = current_val * 0.99 - 0.01
            upper = current_val * 1.01 + 0.01
            if abs(current_val) < 0.01:
                lower = current_val - 0.05
                upper = current_val + 0.05
            
            try:
                result = minimize_scalar(obj_1d_polish, bounds=(lower, upper), method='bounded',
                                        options={'xatol': 1e-14, 'maxiter': 500})
                if result.success and result.fun < best_score:
                    best_coeffs[dim] = result.x
                    best_score = result.fun
            except Exception:
                pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
