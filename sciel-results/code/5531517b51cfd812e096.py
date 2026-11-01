"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_start_time = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 10.0  # seconds

def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients."""
    # c12 determined by g(0) = 0 constraint
    # H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=665280
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    # Use physicists' Hermite polynomials
    # Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # We need a sufficiently large grid
    
    N = 2**15  # 32768 points for FFT
    # x range: need to cover where the function is significant
    # exp(-x^2/2) decays, but H12(x) ~ x^12, so the product ~ x^12 * exp(-x^2/2)
    # This peaks around x ~ sqrt(12) ~ 3.5, so range [-20, 20] should be fine
    x_max = 20.0
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12: use recurrence or explicit formula
    # H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    H2 = 2.0 * x * H1 - 2.0 * H0  # 4x^2 - 2
    H3 = 2.0 * x * H2 - 4.0 * H1
    H4 = 2.0 * x * H3 - 6.0 * H2
    H5 = 2.0 * x * H4 - 8.0 * H3
    H6 = 2.0 * x * H5 - 10.0 * H4
    H7 = 2.0 * x * H6 - 12.0 * H5
    H8 = 2.0 * x * H7 - 14.0 * H6
    H9 = 2.0 * x * H8 - 16.0 * H7
    H10 = 2.0 * x * H9 - 18.0 * H8
    H11 = 2.0 * x * H10 - 20.0 * H9
    H12 = 2.0 * x * H11 - 22.0 * H10
    
    gauss = np.exp(-0.5 * x**2)
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # A(f) = sup |g(x)|
    A_f = np.max(np.abs(g))
    
    # Fourier transform: use FFT
    # Convention: fhat(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # FFT gives: F[k] = sum_j f[j] exp(-2*pi*i*j*k/N)
    # So fhat at frequency xi_k = k/(N*dx) is approximately F[k]*dx
    G = np.fft.fft(g) * dx
    # Shift to center (zero frequency in middle)
    G_shifted = np.fft.fftshift(G)
    
    # A(Fourier(f)) = sup |fhat(xi)|
    A_hat_f = np.max(np.abs(G_shifted))
    
    return A_f * A_hat_f


def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    return _compute_bound(c0, c4, c8)


def construct():
    # Prior state of the art (Goncalves et al. 2017) as listed in the notebook.
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - _SAFETY_MARGIN - elapsed
    
    if remaining <= 5.0:
        return best_coeffs
    
    # Try differential evolution first for global search
    try:
        # Define bounds for the search
        # The seed values are around [-1.13, 0.04, 0.0003]
        # Allow a reasonable range for exploration
        bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
        
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=15,
            polish=True,
            updating='deferred',
            callback=None
        )
        
        if de_result.fun < best_val:
            best_coeffs = de_result.x
            best_val = de_result.fun
        
        elapsed = time.time() - _start_time
        remaining = _DEADLINE - _SAFETY_MARGIN - elapsed
        if remaining <= 5.0:
            return best_coeffs
            
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead or BFGS from best point
    try:
        local_result = minimize(
            _objective,
            x0=best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if local_result.fun < best_val:
            best_coeffs = local_result.x
            best_val = local_result.fun
            
        elapsed = time.time() - _start_time
        remaining = _DEADLINE - _SAFETY_MARGIN - elapsed
        if remaining <= 5.0:
            return best_coeffs
            
    except Exception:
        pass
    
    # Try a few random restarts with local optimization if time permits
    np.random.seed(123)
    for i in range(20):
        elapsed = time.time() - _start_time
        remaining = _DEADLINE - _SAFETY_MARGIN - elapsed
        if remaining <= 10.0:
            break
        
        # Random starting point near the best solution
        scale = [1.0, 0.1, 0.001]
        x0 = best_coeffs + np.random.randn(3) * scale * 0.5
        
        try:
            res = minimize(
                _objective,
                x0=x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_coeffs = res.x
                best_val = res.fun
        except Exception:
            continue
    
    return best_coeffs
# EVOLVE-BLOCK-END
