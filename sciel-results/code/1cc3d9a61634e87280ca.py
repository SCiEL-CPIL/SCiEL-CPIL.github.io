"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.special import factorial

_DEADLINE = time.time() + 195

# Precompute H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 3024.0
_H12_0 = 10321920.0

def _poly_coeffs(a0, a4, a8):
    """Return coefficients of P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 as list [c0,...,c12]."""
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
    # H12(x) = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    coeffs = np.zeros(13)
    # H0
    coeffs[0] += a0
    # H4: 16x^4 - 48x^2 + 12
    coeffs[0] += a4 * 12.0
    coeffs[2] += a4 * (-48.0)
    coeffs[4] += a4 * 16.0
    # H8: 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
    coeffs[0] += a8 * 10080.0
    coeffs[2] += a8 * (-34560.0)
    coeffs[4] += a8 * 23040.0
    coeffs[6] += a8 * (-4608.0)
    coeffs[8] += a8 * 256.0
    # H12: 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
    coeffs[0] += a12 * 10321920.0
    coeffs[2] += a12 * (-34406400.0)
    coeffs[4] += a12 * 34406400.0
    coeffs[6] += a12 * (-12902400.0)
    coeffs[8] += a12 * 2090112.0
    coeffs[10] += a12 * (-147456.0)
    coeffs[12] += a12 * 4096.0
    return coeffs

def _g_function(x, a0, a4, a8):
    """Evaluate g(x) = P(x) * exp(-x^2/2)."""
    coeffs = _poly_coeffs(a0, a4, a8)
    # Evaluate polynomial using Horner's method
    p = np.zeros_like(x)
    for k in range(12, -1, -1):
        p = p * x + coeffs[k]
    return p * np.exp(-x**2 / 2.0)

def _l1_norm(a0, a4, a8, n_points=20000):
    """Compute L1 norm of g(x) = P(x)exp(-x^2/2)."""
    x = np.linspace(-15, 15, n_points)
    g = _g_function(x, a0, a4, a8)
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(np.abs(g), x)
    else:
        return np.trapz(np.abs(g), x)

def _objective(a0, a4, a8):
    """Compute A(g)*A(ĝ) = (1/sqrt(2*pi)) * ||g||_1^2 for the given coefficients."""
    l1 = _l1_norm(a0, a4, a8)
    if not np.isfinite(l1) or l1 < 0:
        return 1e10
    return (1.0 / np.sqrt(2.0 * np.pi)) * l1 * l1

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    best_result = seed_result.copy()
    best_bound = _objective(*seed_result)
    
    # Differential evolution for global search
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    if time.time() < _DEADLINE - 80:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-12,
                polish=True,
                init='sobol',
                popsize=20,
                mutation=(0.3, 1.0),
                recombination=0.8,
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Multi-start Nelder-Mead
    starts = [
        seed_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-0.7, 0.025, 0.00015]),
        np.array([-1.3, 0.055, 0.00045]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B polish
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                _objective,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 200}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with tighter grid
    if time.time() < _DEADLINE - 5:
        def _objective_fine(a0, a4, a8):
            x = np.linspace(-20, 20, 50000)
            coeffs = _poly_coeffs(a0, a4, a8)
            p = np.zeros_like(x)
            for k in range(12, -1, -1):
                p = p * x + coeffs[k]
            g = p * np.exp(-x**2 / 2.0)
            if hasattr(np, 'trapezoid'):
                l1 = np.trapezoid(np.abs(g), x)
            else:
                l1 = np.trapz(np.abs(g), x)
            return (1.0 / np.sqrt(2.0 * np.pi)) * l1 * l1
        
        try:
            res = minimize(
                _objective_fine,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
