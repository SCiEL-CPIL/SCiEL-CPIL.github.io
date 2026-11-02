"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients (physicists' convention: H_n(x) with (-1)^n e^{x^2} d^n/dx^n e^{-x^2})
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 168
# H_12(x) = 4096x^12 - 76800x^10 + 518400x^8 - 1548480x^6 + 2165760x^4 - 1310400x^2 + 10395

def _hermite_coeffs():
    """Return coefficient arrays for H_0, H_4, H_8, H_12 as polynomials in x."""
    # np.poly1d uses descending powers
    h0 = np.array([1.0])
    h4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    h8 = np.array([256.0, 0.0, -3072.0, 0.0, 11520.0, 0.0, -13440.0, 0.0, 168.0])
    h12 = np.array([4096.0, 0.0, -76800.0, 0.0, 518400.0, 0.0, -1548480.0, 0.0,
                    2165760.0, 0.0, -1310400.0, 0.0, 10395.0])
    return h0, h4, h8, h12

def _compute_bound(a, b, c):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function
    g(x) = (a*H_0 + b*H_4 + c*H_8 + d*H_12) * exp(-x^2/2)
    where d is chosen so that g(0) = 0.
    
    The bound involves computing A(g) and A(Fourier(g)) where
    A(h) = sup_x |h(x)| / ||h||_2 (or similar normalized quantity).
    
    Actually, for the uncertainty inequality, we need to compute:
    A(f) = sup_x |f(x)|
    A(Fourier(f)) = sup_x |Fourier(f)(x)|
    
    The product A(f)*A(Fourier(f)) is bounded by a constant times the
    L2 norm squared or similar. Let me think about this more carefully.
    
    For a function g(x) = P(x) * exp(-x^2/2), we can compute:
    - ||g||_2^2 = integral |g(x)|^2 dx
    - A(g) = sup |g(x)|
    - Fourier transform of g
    - A(Fourier(g)) = sup |Fourier(g)(x)|
    
    The bound C4 is such that A(g)*A(Fourier(g)) <= C4 * ||g||_2^2 or similar.
    Actually, the standard form is: A(f) * A(hat{f}) <= C * ||f||_2^2
    
    But we want to minimize the upper bound, which means we want to find
    coefficients that minimize the actual product or a tight upper bound.
    
    Let me compute this numerically.
    """
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # d is determined by g(0) = 0
    # g(0) = (a*H_0(0) + b*H_4(0) + c*H_8(0) + d*H_12(0)) * exp(0)
    # H_0(0) = 1, H_4(0) = 12, H_8(0) = 168, H_12(0) = 10395
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 168.0
    H12_0 = 10395.0
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    # Build the polynomial P(x) = a*H_0 + b*H_4 + c*H_8 + d*H_12
    # We need to add these polynomials
    # Use numpy's polynomial operations
    
    # Create a large enough polynomial array
    max_deg = 12
    P = np.zeros(max_deg + 1)  # P[k] is coefficient of x^k (ascending)
    
    # Convert from descending to ascending
    for coeff, h in zip([a, b, c, d], [h0, h4, h8, h12]):
        deg = len(h) - 1
        for i, c_val in enumerate(h):
            P[deg - i] += coeff * c_val
    
    # Now compute g(x) = P(x) * exp(-x^2/2)
    # We need to find A(g) = sup |g(x)| and A(Fourier(g))
    
    # Numerical approach: sample on a grid
    x = np.linspace(-10, 10, 10000)
    g = np.polyval(P[::-1], x) * np.exp(-x**2 / 2)
    
    A_g = np.max(np.abs(g))
    
    # Compute Fourier transform
    # The Fourier transform convention matters. Let's use the standard:
    # hat{g}(xi) = integral g(x) exp(-2*pi*i*x*xi) dx
    # Or the physics convention: hat{g}(xi) = (1/sqrt(2*pi)) integral g(x) exp(-i*x*xi) dx
    
    # For the uncertainty principle, the convention affects the constant.
    # Let's use the unitary convention: hat{g}(xi) = (1/sqrt(2*pi)) integral g(x) exp(-i*x*xi) dx
    
    # Use FFT for computation
    # First, resample g on a uniform grid suitable for FFT
    N = 2**14  # 16384 points
    x_fft = np.linspace(-20, 20, N, endpoint=False)
    g_fft = np.polyval(P[::-1], x_fft) * np.exp(-x_fft**2 / 2)
    
    dx = x_fft[1] - x_fft[0]
    # FFT gives DFT, we need to scale appropriately
    G_fft = np.fft.fftshift(np.fft.fft(g_fft)) * dx / np.sqrt(2 * np.pi)
    xi_fft = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    A_G = np.max(np.abs(G_fft))
    
    # The product
    product = A_g * A_G
    
    # The bound C4 is this product (or a multiple thereof)
    # We want to minimize this
    return product

def _objective(coeffs):
    """Objective function for optimization."""
    a, b, c = coeffs
    return _compute_bound(a, b, c)

def construct():
    """
    Find three coefficients (a, b, c) for H_0, H_4, H_8 that minimize
    the upper bound on A(f)*A(Fourier(f)).
    The H_12 coefficient is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = _objective(x0)
    best_x = x0.copy()
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    start_points = [x0]
    
    # Add some random perturbations
    for _ in range(20):
        perturbation = rng.randn(3) * 0.5
        start_points.append(x0 + perturbation)
    
    # Also try some structured starting points
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for a_val in [-scale, 0, scale]:
            for b_val in [-scale/2, 0, scale/2]:
                for c_val in [-scale/4, 0, scale/4]:
                    start_points.append(np.array([a_val, b_val, c_val]))
    
    for x_start in start_points:
        if time.time() > deadline - 30:  # Stop if we're running out of time
            break
        
        try:
            result = minimize(_objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < best_result:
                best_result = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Return the best coefficients found
    return best_x
# EVOLVE-BLOCK-END
