"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.integrate import quad
from scipy.special import factorial

_DEADLINE = time.time() + 200

# Precompute H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 3024.0
_H12_0 = 10321920.0

# Precompute Hermite polynomials as coefficient vectors
# H_n(x) coefficients in descending order (x^12, x^11, ..., x^0)
_H0 = np.zeros(13)
_H0[12] = 1.0

_H4 = np.zeros(13)
_H4[8] = 1.0
_H4[6] = -12.0
_H4[4] = 72.0
_H4[2] = -12.0
_H4[0] = 1.0

_H8 = np.zeros(13)
_H8[8] = 1.0
_H8[6] = -56.0
_H8[4] = 1120.0
_H8[2] = -672.0
_H8[0] = 105.0

_H12 = np.zeros(13)
_H12[12] = 1.0
_H12[10] = -66.0
_H12[8] = 2772.0
_H12[6] = -55440.0
_H12[4] = 665280.0
_H12[2] = -399168.0
_H12[0] = 10395.0

def _poly_coeffs(a0, a4, a8):
    """Return coefficients of P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 as list [c0,...,c12]."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    coeffs = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    return coeffs

def _evaluate_poly(coeffs, x):
    """Evaluate polynomial with coefficients in descending order."""
    p = np.zeros_like(x)
    for k in range(13):
        p = p * x + coeffs[k]
    return p

def _find_zeros(coeffs, x_min=-15, x_max=15, n_grid=50000):
    """Find approximate zeros of the polynomial P(x) in [x_min, x_max]."""
    x = np.linspace(x_min, x_max, n_grid)
    p = _evaluate_poly(coeffs, x)
    signs = np.sign(p)
    zeros = []
    for i in range(len(x) - 1):
        if signs[i] != signs[i+1] and signs[i] != 0 and signs[i+1] != 0:
            x0 = x[i] - p[i] * (x[i+1] - x[i]) / (p[i+1] - p[i])
            zeros.append(x0)
    return sorted(set(zeros))

def _l1_norm_fast(a0, a4, a8):
    """Compute L1 norm using fixed quadrature with breakpoints at zeros."""
    coeffs = _poly_coeffs(a0, a4, a8)
    zeros = _find_zeros(coeffs)
    
    breakpoints = [-15.0] + zeros + [15.0]
    breakpoints = sorted(set(breakpoints))
    
    # Precompute Gauss-Legendre nodes
    x_nodes, weights = np.polynomial.legendre.leggauss(96)
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a = breakpoints[i]
        b = breakpoints[i+1]
        if b - a < 1e-14:
            continue
        x_map = 0.5 * (b - a) * x_nodes + 0.5 * (a + b)
        w_map = 0.5 * (b - a) * weights
        p_vals = _evaluate_poly(coeffs, x_map)
        g_vals = np.abs(p_vals * np.exp(-x_map**2 / 2.0))
        total += np.sum(w_map * g_vals)
    return total

def _objective(a0, a4, a8):
    """Compute A(g)*A(ĝ) = (1/sqrt(2*pi)) * ||g||_1^2 for the given coefficients."""
    l1 = _l1_norm_fast(a0, a4, a8)
    if not np.isfinite(l1) or l1 < 0:
        return 1e10
    return (1.0 / np.sqrt(2.0 * np.pi)) * l1 * l1

def _objective_high_precision(a0, a4, a8):
    """High-precision L1 norm using adaptive quadrature."""
    coeffs = _poly_coeffs(a0, a4, a8)
    zeros = _find_zeros(coeffs)
    
    def g_scalar(x):
        p = 0.0
        for k in range(13):
            p = p * x + coeffs[k]
        return abs(p * np.exp(-x**2 / 2.0))
    
    breakpoints = [-15.0] + zeros + [15.0]
    breakpoints = sorted(set(breakpoints))
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a = breakpoints[i]
        b = breakpoints[i+1]
        if b - a < 1e-12:
            continue
        try:
            val, err = quad(g_scalar, a, b, limit=500, epsabs=1e-15, epsrel=1e-15)
            total += val
        except Exception:
            x = np.linspace(a, b, 5000)
            p_vals = _evaluate_poly(coeffs, x)
            g_vals = np.abs(p_vals * np.exp(-x**2 / 2.0))
            if hasattr(np, 'trapezoid'):
                total += np.trapezoid(g_vals, x)
            else:
                total += np.trapz(g_vals, x)
    return (1.0 / np.sqrt(2.0 * np.pi)) * total * total

def _gradients_numerical(a0, a4, a8, h=1e-8):
    """Compute numerical gradient of the objective."""
    f0 = _objective(a0, a4, a8)
    df_da0 = (_objective(a0 + h, a4, a8) - _objective(a0 - h, a4, a8)) / (2*h)
    df_da4 = (_objective(a0, a4 + h, a8) - _objective(a0, a4 - h, a8)) / (2*h)
    df_da8 = (_objective(a0, a4, a8 + h) - _objective(a0, a4, a8 - h)) / (2*h)
    return np.array([df_da0, df_da4, df_da8])

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = seed_result.copy()
    best_bound = _objective(*seed_result)
    
    # Phase 1: Broad differential evolution with Sobol initialization
    bounds = [(-3.0, -0.3), (0.001, 0.15), (1e-5, 0.005)]
    
    if time.time() < _DEADLINE - 100:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=500,
                tol=1e-15,
                polish=True,
                init='sobol',
                popsize=30,
                mutation=(0.4, 1.0),
                recombination=0.9,
                updating='immediate',
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization with diverse starting points
    # Generate starts from a grid around the seed and from DE results
    starts = [
        seed_result,
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.10, 0.038, 0.00030]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.035, 0.00028]),
        np.array([-1.20, 0.045, 0.00035]),
        np.array([-1.00, 0.030, 0.00025]),
        np.array([-1.25, 0.050, 0.00040]),
        np.array([-0.95, 0.025, 0.00020]),
        np.array([-1.30, 0.055, 0.00045]),
        np.array([-1.08, 0.039, 0.00031]),
        np.array([-1.18, 0.043, 0.00033]),
        np.array([-1.12, 0.041, 0.000315]),
        np.array([-1.14, 0.039, 0.000305]),
        np.array([-1.06, 0.037, 0.00029]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Gradient-based polish with L-BFGS-B
    if time.time() < _DEADLINE - 25:
        try:
            res = minimize(
                _objective,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100000, 'ftol': 1e-22, 'gtol': 1e-18, 'maxls': 300}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision verification and polish
    if time.time() < _DEADLINE - 5:
        try:
            hp_bound = _objective_high_precision(*best_result)
            if hp_bound < best_bound:
                best_bound = hp_bound
            
            # One more Nelder-Mead step with high-precision objective
            res = minimize(
                _objective_high_precision,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-17, 'fatol': 1e-19, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
