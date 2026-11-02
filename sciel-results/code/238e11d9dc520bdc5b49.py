"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
import time
import math


def _compute_bound(c0, c4, c8, N=200):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties and Gauss-Hermite quadrature.
    """
    # Coefficients for H_n(0) to determine c12 such that g(0)=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Use Gauss-Hermite quadrature with sufficient points
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    # Compute Hermite polynomials at quadrature points
    x2 = x_gh ** 2
    H0 = np.ones_like(x_gh)
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 4480*x2*x2*x2 + 26880*x2*x2 - 6720*x2 + 1680
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    # g(x) at quadrature points
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x_gh**2 / 2.0)
    
    # A(f) = sup |x * g(x)|
    xg = x_gh * g_vals
    Af = np.max(np.abs(xg))
    
    # The product is sqrt(2*pi) * Af^2
    bound = math.sqrt(2 * math.pi) * Af * Af
    
    return bound


def _compute_bound_grad(coeffs, N=200):
    """Compute bound and gradient using finite differences."""
    c0, c4, c8 = coeffs
    f0 = _compute_bound(c0, c4, c8, N)
    
    h = 1e-8
    dfdc0 = (_compute_bound(c0+h, c4, c8, N) - _compute_bound(c0-h, c4, c8, N)) / (2*h)
    dfdc4 = (_compute_bound(c0, c4+h, c8, N) - _compute_bound(c0, c4-h, c8, N)) / (2*h)
    dfdc8 = (_compute_bound(c0, c4, c8+h, N) - _compute_bound(c0, c4, c8-h, N)) / (2*h)
    
    return f0, np.array([dfdc0, dfdc4, dfdc8])


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    def objective(coeffs):
        try:
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    def objective_with_grad(coeffs):
        try:
            return _compute_bound_grad(coeffs)
        except:
            return 1e10, np.zeros(3)
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Try multiple start points
    start_points = [
        x0,
        np.array([-1.1, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.15, 0.035, 0.0008]),
        np.array([-1.05, 0.045, 0.0012]),
    ]
    
    from scipy.optimize import minimize
    
    # Phase 1: Nelder-Mead to get close to optimum
    for sp in start_points:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 2: L-BFGS-B with gradient for higher precision
    if time.time() < deadline - 20:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', 
                          jac=lambda x: _compute_bound_grad(x)[1],
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Final polish with Nelder-Mead at very tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
