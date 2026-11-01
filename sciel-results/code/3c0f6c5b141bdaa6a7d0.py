"""Optimize Hermite-Gaussian coefficients to minimize the uncertainty bound."""
import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomial coefficients (highest degree first)
_HERMITES = {
    0: np.array([1.0]),
    4: np.array([1.0, 0.0, -6.0, 0.0, 3.0]),
    8: np.array([1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0]),
    12: np.array([1.0, 0.0, -60.0, 0.0, 1260.0, 0.0, -12600.0, 0.0, 56700.0, 0.0, -113400.0, 0.0, 10395.0]),
}

def _compute_g(a0, a1, a2, x):
    """Compute g(x) = (a0*H0 + a1*H4 + a2*H8 + a3*H12) * exp(-x^2/2)"""
    h0 = np.ones_like(x)
    h4 = np.polyval(_HERMITES[4], x)
    h8 = np.polyval(_HERMITES[8], x)
    h12 = np.polyval(_HERMITES[12], x)
    
    # a3 from g(0) = 0: a0 + 3*a1 + 105*a2 + 10395*a3 = 0
    a3 = -(a0 + 3.0*a1 + 105.0*a2) / 10395.0
    
    poly = a0 * h0 + a1 * h4 + a2 * h8 + a3 * h12
    return poly * np.exp(-0.5 * x**2)

def _compute_bound(a0, a1, a2):
    """Compute upper bound on A(f)*A(F(f)) using numerical Fourier transform."""
    # Use a symmetric grid for FFT accuracy
    n = 2048
    L = 20.0
    x = np.linspace(-L, L, n, endpoint=False)
    dx = x[1] - x[0]
    
    g = _compute_g(a0, a1, a2, x)
    max_g = np.max(np.abs(g))
    
    # FFT with proper scaling
    G = np.fft.fftshift(np.fft.fft(g)) * dx
    max_F = np.max(np.abs(G))
    
    return max_g * max_F

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(*x0)
    
    # Multi-start local optimization
    rng = np.random.default_rng(42)
    starts = [x0]
    # Add some random perturbations around initial point
    for _ in range(5):
        pert = rng.normal(0, 0.1, size=3)
        starts.append(x0 + pert)
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(_compute_bound, s, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            final = minimize(_compute_bound, best_x, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if final.fun < best_score:
                best_score = final.fun
                best_x = final.x.copy()
        except Exception:
            pass
    
    return best_x
