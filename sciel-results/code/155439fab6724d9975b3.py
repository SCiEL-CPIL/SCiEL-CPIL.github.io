"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def _hermite_powers(x):
    """Return [H0, H4, H8, H12] evaluated at x."""
    x2 = x**2
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 84*x2**5 + 1260*x2**4 - 11760*x2**3 + 45360*x2**2 - 55440*x2 + 10395
    return H0, H4, H8, H12


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product for given coefficients."""
    # Derive a12 from g(0) = 0 constraint
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
    
    # Use a grid for numerical integration
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    H0, H4, H8, H12 = _hermite_powers(x)
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = P * np.exp(-x**2 / 2.0)
    
    # A(f) = ∫ x²|g(x)|² dx
    A_f = np.trapz(x**2 * g**2, x)
    
    # Fourier transform of g(x) = P(x)*exp(-x²/2)
    # Use FFT for the Fourier transform
    # Pad to power of 2 for FFT
    n = len(x)
    nfft = 1 << (n - 1).bit_length()
    g_fft = np.fft.fft(g, n=nfft) * dx
    
    # Frequency axis
    dx_freq = 1.0 / (nfft * dx)
    freq = np.fft.fftfreq(nfft, d=dx)
    
    # A(F(f)) = ∫ ξ²|F(g)(ξ)|² dξ
    A_F = np.trapz(freq**2 * np.abs(g_fft)**2, dx=dx_freq)
    
    return A_f * A_F


def _objective_spherical(theta, phi):
    """Objective in spherical coordinates. Normalizes the coefficient vector."""
    # Convert spherical to Cartesian in 3D
    # We use 2 angles since the vector is normalized
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    
    return _compute_bound(a0, a4, a8)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Start with the known good solution
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_score = _compute_bound(*best_coeffs)
    
    # Use Differential Evolution in spherical coordinates
    # theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi), (0, 2*np.pi)]
    
    try:
        result = differential_evolution(
            _objective_spherical,
            bounds=bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True,
            callback=None
        )
        
        elapsed = time.time() - start_time
        if result.success and result.fun < best_score:
            theta, phi = result.x
            a0 = np.sin(theta) * np.cos(phi)
            a4 = np.sin(theta) * np.sin(phi)
            a8 = np.cos(theta)
            best_coeffs = np.array([a0, a4, a8])
            best_score = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 160:
        # Return what we have
        return best_coeffs
    
    # Try a few more random restarts if we have time
    for seed in [1, 2, 3, 7, 13]:
        if time.time() > deadline - 20:
            break
        try:
            result = differential_evolution(
                _objective_spherical,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=seed,
                polish=True
            )
            if result.fun < best_score:
                theta, phi = result.x
                a0 = np.sin(theta) * np.cos(phi)
                a4 = np.sin(theta) * np.sin(phi)
                a8 = np.cos(theta)
                best_coeffs = np.array([a0, a4, a8])
                best_score = result.fun
        except Exception:
            continue
    
    return best_coeffs
# EVOLVE-BLOCK-END
