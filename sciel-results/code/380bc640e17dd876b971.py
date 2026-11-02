import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(1, n):
        h2 = 2 * x * h1 - 2 * k * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound C4 for given coefficients.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    with a12 chosen so g(0) = 0.
    The bound is A(f)*A(F) / ||f||_2^2 where A(f) = sqrt(int x^2 |f|^2 dx)
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    x = np.linspace(-20, 20, 50000)
    gauss = np.exp(-x**2 / 2)
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    g2 = g**2
    
    norm2 = np.trapz(g2, x)
    if norm2 <= 1e-30:
        return 1e10
    
    A2 = np.trapz(x**2 * g2, x)
    
    # For Fourier-invariant function, A(F) = A(f), so product = A2
    # Bound = A2 / norm2 (this is the variance-like quantity)
    # Or maybe bound = A2^2 / norm2^2
    # Let's use A2 / norm2 as the spread squared
    
    bound = A2 / norm2
    return bound

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin
    
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _compute_bound(*x0)
    
    start_points = [
        x0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0005]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([0.0, 0.05, 0.0003]),
        np.array([-1.13, 0.04, 0.0003]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(_compute_bound, sp, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish
    if time.time() < deadline - 30:
        try:
            res = minimize(_compute_bound, best_x, method='L-BFGS-B',
                          options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
