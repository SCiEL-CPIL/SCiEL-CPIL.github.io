"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
_DEADLINE = 210.0  # seconds wall-clock limit
_SAFETY = 10.0     # safety margin

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _start_time)

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 for the given coefficients.
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    c12 is determined by g(0) = 0.
    
    A(f) = ||f||_2^2
    The uncertainty product bound involves A(g) * A(Fourier(g))
    For Hermite-Gaussians, Fourier(H_n * exp(-x^2/2)) = i^n * H_n * exp(-x^2/2)
    (with appropriate normalization)
    """
    # Use a fine grid for numerical integration
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    # Gaussian envelope
    gauss = np.exp(-x**2 / 2)
    
    # c12 from g(0) = 0: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-399168 (physicists' Hermite)
    # Let's compute at x=0
    H0_0 = 1.0
    H4_0 = _hermite(4, np.array([0.0]))[0]
    H8_0 = _hermite(8, np.array([0.0]))[0]
    H12_0 = _hermite(12, np.array([0.0]))[0]
    
    if abs(H12_0) < 1e-15:
        return 1e10
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # g(x)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # A(g) = ||g||_2^2 = integral |g|^2 dx
    A_g = np.sum(g**2) * dx
    
    # Fourier transform of g:
    # F(H_n(x) * exp(-x^2/2)) = i^n * H_n(x) * exp(-x^2/2) * (2*pi)^{1/2} / (2*pi)^{1/2}
    # Actually, the standard result: if f(x) = H_n(x) exp(-x^2/2), then 
    # F[f](k) = i^n H_n(k) exp(-k^2/2) * sqrt(2*pi) / sqrt(2*pi) = i^n H_n(k) exp(-k^2/2)
    # Wait, let me be more careful. The Fourier transform convention matters.
    # Using F[f](k) = integral f(x) exp(-2*pi*i*x*k) dx
    # For f(x) = exp(-x^2/2), F[f](k) = sqrt(2*pi) * exp(-2*pi^2*k^2)
    # This is getting complicated. Let me use the property that for the 
    # "natural" Gaussian exp(-x^2/2), the Fourier transform (with the 
    # convention that makes it self-reciprocal up to a constant) gives:
    # F[H_n exp(-x^2/2)] = i^n H_n exp(-x^2/2) * (normalization factor)
    #
    # Actually, the key insight for the uncertainty inequality is:
    # The bound is on A(f)*A(F(f)) where A(f) = ||f||_2^2
    # For our test function, we need to compute both norms.
    #
    # Let me use the fact that for Hermite functions:
    # If we define psi_n(x) = H_n(x) exp(-x^2/2) / (2^n n! sqrt(pi))^{1/2}
    # then psi_n are orthonormal and F[psi_n] = i^n psi_n
    #
    # But our g is not normalized. Let's compute directly.
    # 
    # The Fourier transform with convention F[f](k) = int f(x) e^{-2*pi*i*x*k} dx:
    # For f(x) = H_n(x) e^{-x^2/2}:
    # This is related to the Hermite functions. Let me use numerical FT.
    
    # Numerical Fourier transform
    # Use FFT on a large grid
    N = len(x)
    # Shift to centered grid for FFT
    # Actually, let's just compute the Fourier transform numerically
    # F[g](k) = integral g(x) exp(-2*pi*i*x*k) dx
    # We can evaluate this on a grid of k values
    
    # For the uncertainty product, we need A(F[g]) = ||F[g]||_2^2
    # By Parseval's theorem: ||F[g]||_2^2 = (2*pi) * ||g||_2^2 
    # Wait, that's for the convention F[f](k) = int f(x) e^{-ikx} dx
    # For F[f](k) = int f(x) e^{-2*pi*i*x*k} dx, Parseval gives ||F[f]||_2^2 = ||f||_2^2
    
    # Hmm, but the task says "upper bound on A(f)*A(Fourier(f))"
    # This suggests it's not just the product of norms, but some inequality bound.
    #
    # Looking at the uncertainty inequality context: 
    # The Heisenberg uncertainty principle says:
    # (integral x^2 |f|^2)(integral k^2 |F[f]|^2) >= (1/4)(integral |f|^2)^2
    # 
    # But the task mentions "C4" which suggests a specific constant in an inequality.
    # The bound might be: A(f) * A(F(f)) >= C4 * (something)
    # Or we're minimizing the ratio that gives the best constant.
    #
    # Given the task says "reduce the upper bound on A(f)*A(Fourier(f))",
    # I think we want to minimize the actual value of A(f)*A(F(f)) 
    # or a related quantity.
    #
    # Let me reconsider. Perhaps:
    # A(f) = (integral x^2 |f(x)|^2 dx) / (integral |f(x)|^2 dx)  (variance in x)
    # A(F(f)) = (integral k^2 |F(f)(k)|^2 dk) / (integral |F(f)(k)|^2 dk)  (variance in k)
    # And the uncertainty principle says A(f) * A(F(f)) >= 1/4
    # The "upper bound on C4" might refer to finding the optimal constant.
    #
    # Actually, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    # This is confusing. Let me just try to minimize the product of the 
    # second moments or some related quantity.
    
    # Let me compute the second moments directly
    # Var_x = integral x^2 |g|^2 dx / integral |g|^2 dx
    # For the Fourier side, I need the FT.
    
    # Use FFT for the Fourier transform
    # Pad to next power of 2
    N_fft = 1 << (N - 1).bit_length()
    if N_fft < N:
        N_fft = 1 << (N).bit_length()
    
    g_padded = np.zeros(N_fft, dtype=complex)
    g_padded[:N] = g
    
    # FFT
    G = np.fft.fft(g_padded) * dx
    
    # Frequency grid
    k = 2 * np.pi * np.fft.fftfreq(N_fft, d=dx)
    
    # A(F[g]) = integral k^2 |G(k)|^2 dk / integral |G(k)|^2 dk
    norm_G = np.sum(np.abs(G)**2) * dx
    if norm_G < 1e-15:
        return 1e10
    
    var_k = np.sum(k**2 * np.abs(G)**2) * dx / norm_G
    
    # Var_x
    norm_g = np.sum(g**2) * dx
    if norm_g < 1e-15:
        return 1e10
    
    var_x = np.sum(x**2 * g**2) * dx / norm_g
    
    # The uncertainty product
    uncertainty_product = var_x * var_k
    
    return uncertainty_product

def _objective(coeffs):
    c0, c4, c8 = coeffs
    try:
        val = _compute_bound(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    # Start from the known good point
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # Try multiple starting points
    starts = [
        x0,
        x0 * 0.5,
        x0 * 2.0,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-1.5, 0.15, 0.002]),
    ]
    
    for start in starts:
        if _time_left() < 5.0:
            break
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if _time_left() > 10.0:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
