"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

def _hermite_poly(n, x):
    """Physicists' Hermite polynomial H_n(x) evaluated at x."""
    H = hermite(n)
    return H(x)

def _compute_bound(a, b, c):
    """
    Compute C4 = (||f||₄ / ||f||₂)² where f = P * e^{-x²/2}
    P(x) = a*H0 + b*H4 + c*H8 + d*H12, d chosen so P(0) = 0.
    """
    # Find d so P(0) = 0
    H0_0 = _hermite_poly(0, np.array([0.0]))[0]
    H4_0 = _hermite_poly(4, np.array([0.0]))[0]
    H8_0 = _hermite_poly(8, np.array([0.0]))[0]
    H12_0 = _hermite_poly(12, np.array([0.0]))[0]
    
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0
    
    # Numerical integration
    x = np.linspace(-12, 12, 15001)
    dx = x[1] - x[0]
    
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    P = a * H0 + b * H4 + c * H8 + d * H12
    f = P * np.exp(-0.5 * x**2)
    
    norm2 = np.sum(f**2) * dx
    norm4 = np.sum(f**4) * dx
    
    if norm2 <= 0 or norm4 <= 0:
        return 1e10
    
    C4 = (norm4 / norm2**2)
    
    return C4

def construct():
    start_time = time.time()
    deadline = start_time + 195
    
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # BFGS
    try:
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 5000, 'gtol': 1e-12})
        if res.fun < best_bound:
            best_bound = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time > deadline - 40:
        return best_x
    
    # Multi-start
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > deadline - 20:
            break
        scale = 0.05 + 0.05 * trial
        x_trial = x0 + np.random.randn(3) * scale
        try:
            res = minimize(objective, x_trial, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11})
            if res.fun < best_bound:
                best_bound = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Polish
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(objective, best_x, method='BFGS',
                          options={'maxiter': 10000, 'gtol': 1e-14})
            if res.fun < best_bound:
                best_bound = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END

