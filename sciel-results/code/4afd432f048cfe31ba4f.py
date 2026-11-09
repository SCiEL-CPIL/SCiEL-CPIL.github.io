"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# Precompute grid and Fourier-related constants
_NPTS = 2048
_XMAX = 8.0
_x_grid = np.linspace(-_XMAX, _XMAX, _NPTS)
_dx = _x_grid[1] - _x_grid[0]
# Fourier grid (using convention: F(ξ) = ∫ f(x) e^{-2πiξx} dx)
_ξ_grid = np.fft.fftfreq(_NPTS, d=_dx)
_2πi = 2j * np.pi
# Precompute exp(-x^2)
_gauss = np.exp(-_x_grid**2)


def _hermite_poly_coeffs(deg):
    """Return coefficients of physicists' Hermite polynomial H_n (highest power first)."""
    if deg == 0:
        return np.array([1.0])
    if deg == 4:
        return np.array([16.0, 0.0, -48.0, 0.0, 12.0])
    if deg == 8:
        return np.array([256.0, 0.0, -4480.0, 0.0, 26880.0, 0.0, -33600.0, 0.0, 1680.0])
    if deg == 12:
        return np.array([4096.0, 0.0, -245760.0, 0.0, 6635520.0, 0.0,
                         -77414400.0, 0.0, 405504000.0, 0.0,
                         -798336000.0, 0.0, 4790016.0])
    raise ValueError(f"Unsupported degree {deg}")


# Hermite values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 4790016.0


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound C4 for given Hermite coefficients.
    
    The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2)
    where a12 is chosen so that g(0) = 0.
    
    The bound C4 is computed as the product of uncertainties:
    C4 = (∫ x^2 |g|^2 dx / ∫|g|^2 dx) * (∫ ξ^2 |ĝ|^2 dξ / ∫|ĝ|^2 dξ)
    which is the Heisenberg uncertainty product for the test function.
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Evaluate polynomial on grid
    x = _x_grid
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4480x^6 + 26880x^4 - 33600x^2 + 1680
    # H12 = 4096x^12 - 245760x^10 + 6635520x^8 - 77414400x^6 + 405504000x^4 - 798336000x^2 + 4790016
    
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 33600.0*x2 + 1680.0
    H12 = 4096.0*x12 - 245760.0*x10 + 6635520.0*x8 - 77414400.0*x6 + 405504000.0*x4 - 798336000.0*x2 + 4790016.0
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = P * _gauss
    
    # Position side
    g2 = g**2
    norm2 = np.sum(g2) * _dx
    if norm2 < 1e-30:
        return 1e10
    x2_weighted = np.sum(x2 * g2) * _dx
    var_x = x2_weighted / norm2
    
    # Fourier side: use FFT
    # g_hat[ξ] ≈ ∫ g(x) e^{-2πiξx} dx ≈ sum g(x_j) e^{-2πiξ_k x_j} dx
    # Using FFT: if we define f_k = g(x_k), then FFT gives sum f_j e^{-2πi jk/N}
    # We need to account for the non-uniform phase in x
    # Better: use the fact that on a uniform grid, FFT approximates the DFT
    # g_hat_k ≈ dx * sum_j g_j * exp(-2πi * ξ_k * x_j)
    # Since ξ_k = k/(N*dx) and x_j = -XMAX + j*dx, we have:
    # exp(-2πi * k/(N*dx) * (-XMAX + j*dx)) = exp(2πi*k*XMAX/(N*dx)) * exp(-2πi*k*j/N)
    # The first factor is a phase shift that doesn't affect |g_hat|^2
    # So |g_hat_k|^2 ≈ dx^2 * |FFT(g)_k|^2 (up to the phase)
    
    g_hat = np.fft.fft(g) * _dx
    g_hat2 = np.abs(g_hat)**2
    
    norm2_f = np.sum(g_hat2)
    if norm2_f < 1e-30:
        return 1e10
    xi2_weighted = np.sum(_ξ_grid**2 * g_hat2)
    var_ξ = xi2_weighted / norm2_f
    
    # The uncertainty product
    C4 = var_x * var_ξ
    
    return C4


def construct():
    """
    Find optimal Hermite coefficients to minimize the upper bound C4.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s kill
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple starting points
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    # Multi-start optimization with various strategies
    np.random.seed(42)
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() > deadline - 10:  # Leave 10s safety margin
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # Try symmetric perturbation
            x_start = x0 + np.array([0.1, -0.01, 0.001])
        elif i == 2:
            x_start = x0 + np.array([-0.1, 0.01, -0.001])
        elif i == 3:
            # Larger perturbation
            x_start = x0 + np.random.randn(3) * 0.5
        elif i == 4:
            x_start = x0 + np.random.randn(3) * 0.2
        elif i == 5:
            x_start = x0 + np.random.randn(3) * 0.05
        else:
            # Random restarts with varying scale
            scale = 0.01 * (10 ** (i / 5.0))
            x_start = x0 + np.random.randn(3) * scale
        
        try:
            result = minimize(
                _compute_bound,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
                
                # If we found a good point, do a local refinement
                if time.time() < deadline - 30:
                    try:
                        result2 = minimize(
                            _compute_bound,
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
                        )
                        if result2.fun < best_bound:
                            best_bound = result2.fun
                            best_x = result2.x.copy()
                    except Exception:
                        pass
        except Exception:
            continue
    
    return best_x
# EVOLVE-BLOCK-END
