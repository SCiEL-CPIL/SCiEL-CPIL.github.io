"""Fourier-invariant Hermite-Gaussian test function tuned via spectral moment optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial


def _hermite_poly(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_nm2 = np.ones_like(x)
        h_nm1 = 2.0 * x
        for k in range(2, n + 1):
            h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
            h_nm2 = h_nm1
            h_nm1 = h_n
        return h_nm1


def _compute_uncertainty_product(a_coeffs, grid=None):
    """
    Compute the uncertainty product A(f)*A(F) for g(x) = sum a_k H_k(x) e^{-x^2/2}.
    a_coeffs = [a0, a4, a8, a12] where a12 is determined by g(0)=0.
    Returns the product sqrt(<x^2>) * sqrt(<xi^2>) = sqrt(<x^2><xi^2>).
    """
    if grid is None:
        # Use a sufficiently fine grid
        x = np.linspace(-15, 15, 20001)
        dx = x[1] - x[0]
        grid = (x, dx)
    x, dx = grid
    
    a0, a4, a8, a12 = a_coeffs
    
    # Compute g(x)
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    gaussian = np.exp(-0.5 * x**2)
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gaussian
    
    # Compute |g|^2
    g_sq = g**2
    
    # Normalize
    norm_sq = np.sum(g_sq) * dx
    if norm_sq < 1e-30:
        return 1e10  # invalid
    
    prob_x = g_sq / norm_sq
    
    # <x^2>
    x2 = np.sum(x**2 * prob_x) * dx
    
    # Compute Fourier transform using FFT for accuracy
    # Use a larger grid for FFT
    N_fft = 2**16  # 65536
    x_fft = np.linspace(-15, 15, N_fft)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f = _hermite_poly(0, x_fft)
    H4_f = _hermite_poly(4, x_fft)
    H8_f = _hermite_poly(8, x_fft)
    H12_f = _hermite_poly(12, x_fft)
    
    gaussian_f = np.exp(-0.5 * x_fft**2)
    g_f = (a0 * H0_f + a4 * H4_f + a8 * H8_f + a12 * H12_f) * gaussian_f
    
    # FFT to get Fourier transform
    G_f = np.fft.fftshift(np.fft.fft(g_f)) * dx_fft
    G_f_sq = np.abs(G_f)**2
    
    norm_f_sq = np.sum(G_f_sq) * dx_fft
    if norm_f_sq < 1e-30:
        return 1e10
    
    prob_k = G_f_sq / norm_f_sq
    
    # <k^2> where k = 2*pi*f, and f = fftfreq
    freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dx_fft))
    k_vals = 2.0 * np.pi * freq
    k2 = np.sum(k_vals**2 * prob_k) * dx_fft
    
    # Uncertainty product
    prod = np.sqrt(x2 * k2)
    return prod


def _objective(a3):
    """Objective function for optimization. a3 = [a0, a4, a8]."""
    a0, a4, a8 = a3
    # a12 from g(0) = 0: a0 + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H4(0) = -12, H8(0) = 1680, H12(0) = -47880
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 47880.0
    a_full = np.array([a0, a4, a8, a12])
    return _compute_uncertainty_product(a_full)


def construct():
    start = time.time()
    deadline = start + 175.0  # safety margin before 210s kill
    
    # Precompute grid for speed
    x_grid = np.linspace(-15, 15, 20001)
    dx_grid = x_grid[1] - x_grid[0]
    grid = (x_grid, dx_grid)
    
    # Seed value
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best = seed.copy()
    best_val = _objective(best)
    
    # Try multiple starting points with different strategies
    start_points = [
        seed,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.1, 0.01]),
        np.array([1.0, -0.1, 0.01]),
        np.array([-0.5, 0.5, -0.01]),
        np.array([0.5, -0.5, 0.02]),
    ]
    
    # Also try scaled versions
    for s in [0.1, 0.5, 2.0, 5.0]:
        start_points.append(seed * s)
    
    # Try gradient-based optimization from each starting point
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success or res.fun < best_val:
                if np.all(np.isfinite(res.x)):
                    best = res.x
                    best_val = res.fun
        except Exception:
            pass
    
    # If we still have time, try more refined search
    if time.time() < deadline - 10:
        # Do a local refinement around the best point
        try:
            res = minimize(_objective, best, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                best = res.x
                best_val = res.fun
        except Exception:
            pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best)):
        best = seed
    
    return best
# EVOLVE-BLOCK-END
