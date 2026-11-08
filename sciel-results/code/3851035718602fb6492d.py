"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_hermite

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 175.0  # seconds, with safety margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # H_k(0) for even k: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-103950
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1680.0
    h12_0 = -103950.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12

# Precompute Gauss-Hermite quadrature nodes and weights
# For integral of f(x) * exp(-x^2) dx, use roots_hermite(n)
# We need integral of |g(x)| dx where g(x) = P(x) * exp(-x^2/2)
# So |g(x)| = |P(x)| * exp(-x^2/2)
# Let u = x/sqrt(2), then dx = sqrt(2) du, exp(-x^2/2) = exp(-u^2)
# Integral = sqrt(2) * integral |P(sqrt(2)*u)| * exp(-u^2) du
# Use Gauss-Hermite with n=200 for high accuracy
_N_QUAD = 200
_x_nodes, _w_nodes = roots_hermite(_N_QUAD)
# Transform: x = sqrt(2) * u, so we evaluate P at sqrt(2)*u
_x_transformed = np.sqrt(2.0) * _x_nodes
_w_transformed = np.sqrt(2.0) * _w_nodes

def _hermite_poly(x, k):
    """Compute Hermite polynomial H_k(x) for even k in {0, 4, 8, 12}."""
    x2 = x * x
    if k == 0:
        return np.ones_like(x) if np.ndim(x) > 0 else 1.0
    elif k == 4:
        return 16*x2**2 - 48*x2 + 12
    elif k == 8:
        return 256*x2**4 - 3072*x2**3 + 11520*x2**2 - 13440*x2 + 1680
    elif k == 12:
        return 4096*x2**6 - 76800*x2**5 + 518400*x2**4 - 1548480*x2**3 + 2342400*x2**2 - 1548480*x2 + 103950
    else:
        raise ValueError(f"Unsupported Hermite degree: {k}")

def _compute_bound_fast(c0, c4, c8):
    """Compute the upper bound using precomputed Gauss-Hermite quadrature."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Evaluate polynomial part at quadrature nodes
    H0 = _hermite_poly(_x_transformed, 0)
    H4 = _hermite_poly(_x_transformed, 4)
    H8 = _hermite_poly(_x_transformed, 8)
    H12 = _hermite_poly(_x_transformed, 12)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # |g(x)| = |P(x)| * exp(-x^2/2)
    # Using the transformed variables, integral = sum(w_i * |P(x_i)|)
    # where x_i are already transformed
    abs_P = np.abs(P)
    A_f = np.sum(_w_transformed * abs_P)
    
    # A_F = sqrt(2*pi) * A_f
    # Product = sqrt(2*pi) * A_f^2
    A_F = np.sqrt(2 * np.pi) * A_f
    return A_f * A_F

def _compute_bound_analytic(c0, c4, c8):
    """Compute the upper bound - uses fast Gauss-Hermite method."""
    return _compute_bound_fast(c0, c4, c8)

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_fast,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try multi-start with different initial points
    if _time_left() > 50:
        try:
            # Try several random starting points
            np.random.seed(42)
            for _ in range(10):
                if _time_left() < 20:
                    break
                start = np.random.uniform([-5, 5, -0.5, 0.5], [5, 5, 0.5, -0.5])[:3]
                res = minimize(
                    _compute_bound_fast,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 80:
        try:
            # Reasonable bounds for coefficients
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            res_de = differential_evolution(
                _compute_bound_fast,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-8,
                polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Final high-precision polish pass
    if _time_left() > 15 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
