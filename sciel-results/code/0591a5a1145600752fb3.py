"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute the n-th probabilists' Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    H0 = np.ones_like(x)
    H1 = x
    for k in range(2, n + 1):
        H2 = x * H1 - (k - 1) * H0
        H0, H1 = H1, H2
    return H1

def _compute_objective(coeffs):
    """Compute the upper bound on A(f)*A(Fourier(f)) for given (a0, a4, a8)."""
    a0, a4, a8 = coeffs
    # a12 from g(0) = 0: a0 + 3*a4 + 105*a8 + 10395*a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
    
    # Use a fine grid for numerical integration
    x = np.linspace(-20, 20, 200001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = np.ones_like(x)
    H1 = x
    H2 = x * H1 - H0
    H3 = x * H2 - 2 * H1
    H4 = x * H3 - 3 * H2
    H5 = x * H4 - 4 * H3
    H6 = x * H5 - 5 * H4
    H7 = x * H6 - 6 * H5
    H8 = x * H7 - 7 * H6
    H9 = x * H8 - 8 * H7
    H10 = x * H9 - 9 * H8
    H11 = x * H10 - 10 * H9
    H12 = x * H11 - 11 * H10
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-0.5 * x**2)
    
    # A(g) = integral of |g(x)| dx
    Ag = np.sum(np.abs(g)) * dx
    
    # Fourier transform: for g(x) = P(x)*exp(-x^2/2), 
    # F(g)(xi) = (1/sqrt(2*pi)) * integral g(x) exp(-i*x*xi) dx
    # We can compute this via FFT or direct integration
    # Using the fact that F[exp(-x^2/2)] = exp(-xi^2/2) (up to normalization)
    # Let's use numerical FFT approach
    
    # Actually, let's use a simpler approach: compute the Fourier transform numerically
    # F(g)(xi) = (1/sqrt(2*pi)) * int g(x) e^{-ixi x} dx
    
    # For efficiency, use the same grid
    xi = x  # symmetric grid
    # Compute 2D: for each xi, integrate g(x)*exp(-i*xi*x)
    # This is expensive, so let's use FFT
    
    # Use FFT: need uniform grid
    # x goes from -20 to 20 with 200001 points
    # FFT gives us the discrete Fourier transform
    
    # Normalize for proper scaling
    N = len(x)
    # The FFT: sum_k g[k] * exp(-2*pi*i*k*n/N)
    # Our integral: int g(x) exp(-i*xi*x) dx ≈ dx * sum g[k] exp(-i*xi_k * x_k)
    
    # The FFT frequencies: xi_k = 2*pi*k/(N*dx) for k=0,...,N/2, then negative
    # Actually let's just do it more carefully
    
    # Use scipy's fft
    G_fft = np.fft.fft(g) * dx
    # Shift so zero frequency is in the middle
    G_fft = np.fft.fftshift(G_fft)
    
    # Frequency axis: xi_k = 2*pi*(k - N/2)/(N*dx)
    xi_freq = 2 * np.pi * (np.arange(N) - N // 2) / (N * dx)
    
    # A(g_hat) = (1/sqrt(2*pi)) * integral |F(g)(xi)| dxi
    # F(g)(xi) = (1/sqrt(2*pi)) * integral g(x) exp(-i*xi*x) dx
    # So |F(g)(xi)| = (1/sqrt(2*pi)) * |integral g(x) exp(-i*xi*x) dx|
    # A(g_hat) = (1/sqrt(2*pi)) * integral |F(g)(xi)| dxi
    #           = (1/sqrt(2*pi)) * integral (1/sqrt(2*pi)) |int g(x) e^{-ixi x} dx| dxi
    #           = (1/(2*pi)) * integral |int g(x) e^{-ixi x} dx| dxi
    
    # The integral |int g(x) e^{-ixi x} dx| dxi ≈ dx * sum |G_fft[k]|
    # But we need to be careful about the normalization
    
    # Actually, let me reconsider. The standard Fourier transform:
    # F(g)(xi) = (1/sqrt(2*pi)) * int g(x) e^{-i*xi*x} dx
    # A(F(g)) = int |F(g)(xi)| dxi = (1/sqrt(2*pi)) * int |int g(x) e^{-ixi x} dx| dxi
    
    # The inner integral approximated by dx * sum g[k] e^{-i xi_k x_k}
    # which is what FFT gives us (with appropriate scaling)
    
    # G_fft[k] = dx * sum_j g[j] exp(-2*pi*i*j*k/N)
    # And xi_k = 2*pi*(k-N/2)/(N*dx) after shift
    # So G_fft[k] ≈ int g(x) exp(-i*xi_k*x) dx
    
    # Therefore A(F(g)) = (1/sqrt(2*pi)) * dx * sum |G_fft[k]|
    # Wait, no. G_fft already includes dx.
    # int |int g(x) e^{-ixi x} dx| dxi ≈ dx * sum_k |G_fft[k]|
    
    A_Fg = (1.0 / np.sqrt(2 * np.pi)) * dx * np.sum(np.abs(G_fft))
    
    return Ag * A_Fg

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # safety margin before 210s limit
    
    # Initial guess from prior art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple optimization strategies
    best_val = _compute_objective(x0)
    best_x = x0.copy()
    
    # Strategy 1: Standard BFGS/L-BFGS-B optimization
    try:
        result = minimize(
            _compute_objective, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 2: Try different initial points
    initial_points = [
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.5, 0.06, 0.0001]),
        np.array([-1.0, 0.05, 0.0002]),
        np.array([-0.9, 0.035, 0.00035]),
    ]
    
    for ip in initial_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _compute_objective, ip, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 3: Polynomial root-finding for critical points
    # The objective is a function of 3 variables. At a critical point, 
    # the gradient is zero. For a polynomial objective, this gives 
    # polynomial equations. Let's try to find these.
    # 
    # However, the objective involves L1 norms which are piecewise linear,
    # making it non-smooth. So pure polynomial root-finding won't work directly.
    # Instead, let's do a finer local search around the best point found.
    
    # Strategy 4: Nelder-Mead for local refinement
    try:
        result = minimize(
            _compute_objective, best_x, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 5: Even finer polish with COBYLA
    try:
        result = minimize(
            _compute_objective, best_x, method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 1e-6}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
