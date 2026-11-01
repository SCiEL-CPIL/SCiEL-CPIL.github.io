import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 180.0  # seconds, with safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return _DEADLINE - _elapsed()

def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x) if hasattr(x, '__len__') else 1.0
    elif n == 1:
        return 2 * x
    # Use recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    if isinstance(x, (int, float, np.floating)):
        h0 = 1.0
        h1 = 2.0 * x
        if n == 0:
            return h0
        if n == 1:
            return h1
        for k in range(1, n):
            h0, h1 = h1, 2.0 * x * h1 - 2.0 * k * h0
        return h1
    else:
        x = np.asarray(x)
        h0 = np.ones_like(x)
        h1 = 2.0 * x
        if n == 0:
            return h0
        if n == 1:
            return h1
        for k in range(1, n):
            h0, h1 = h1, 2.0 * x * h1 - 2.0 * k * h0
        return h1

def _g_of_x(x, coeffs):
    """Compute g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)."""
    a0, a4, a8 = coeffs
    # a12 is determined by g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 1680.0) / 2032128.0
    
    h0 = _hermite_poly(0, x)
    h4 = _hermite_poly(4, x)
    h8 = _hermite_poly(8, x)
    h12 = _hermite_poly(12, x)
    
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-x**2 / 2.0)

def _A_g(coeffs):
    """Compute A(g) = ∫|g(x)|dx."""
    f = lambda x: abs(_g_of_x(x, coeffs))
    # Integrate from -inf to inf; use symmetry? g is even, so integrate 0 to inf and double
    val, _ = quad(f, 0, np.inf, limit=200)
    return 2.0 * val

def _fourier_g(x, coeffs):
    """Compute F(g)(x) = (1/sqrt(2*pi)) * ∫ g(t) e^(-itx) dt."""
    # Use the fact that g is real and even, so F(g) is real and even
    # F(g)(x) = (1/sqrt(2*pi)) * 2 * ∫_0^∞ g(t) cos(tx) dt
    return (2.0 / np.sqrt(2.0 * np.pi)) * quad(lambda t: _g_of_x(t, coeffs) * np.cos(t * x), 0, np.inf, limit=200)[0]

def _A_Fg(coeffs):
    """Compute A(F(g)) = ∫|F(g)(ξ)|dξ."""
    f = lambda xi: abs(_fourier_g(xi, coeffs))
    val, _ = quad(f, 0, np.inf, limit=200)
    return 2.0 * val

def _objective(coeffs):
    """Compute A(g) * A(F(g))."""
    try:
        ag = _A_g(coeffs)
        afg = _A_Fg(coeffs)
        return ag * afg
    except Exception:
        return np.inf

def construct():
    # Initial guess from prior work
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try to improve with scipy.optimize
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Check time and return
    if _time_remaining() > 5.0:
        # Try a few random restarts if time permits
        rng = np.random.RandomState(42)
        for _ in range(5):
            if _time_remaining() < 10.0:
                break
            x_try = best_x + rng.normal(0, 0.1, size=3)
            val = _objective(x_try)
            if val < best_val:
                best_x = x_try
                best_val = val
    
    return best_x
