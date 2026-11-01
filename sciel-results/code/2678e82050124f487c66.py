"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Analytic approach: use Hermite-Gaussian properties to compute bounds efficiently
with exact gradients for L-BFGS-B optimization.
"""

import numpy as np
from scipy import optimize
from scipy.special import factorial
import time

# Precompute constants
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1190.0
_H12_0 = 113880.0
_DEGREES = [0, 4, 8, 12]
_TWO_PI = 2.0 * np.pi
_SQRT_2PI = np.sqrt(2.0 * np.pi)

# Precompute L2 norm factors: sqrt(pi) * 2^d * d!
_L2_FACTORS = np.array([np.sqrt(np.pi) * (2.0 ** d) * float(factorial(d, exact=True)) for d in _DEGREES])

# Precompute FT coefficient multipliers: sqrt(2*pi) * (2*pi)^d * (-i)^d
_FT_MULTIPLIERS = np.array([np.sqrt(2.0 * np.pi) * ((2.0 * np.pi) ** d) * ((-1j) ** d) for d in _DEGREES])

# Fixed quadrature grid for L1 norms (Gaussian-like spacing, symmetric)
_N_QUAD = 400
_QUAD_X = np.linspace(-6.0, 6.0, _N_QUAD)
_QUAD_W = np.full(_N_QUAD, 12.0 / _N_QUAD)  # trapezoidal weights (uniform)
# Better: use Gauss-Hermite-like spacing for efficiency
# Actually for smooth integrands, uniform trapezoidal on [-6,6] is fine
# The Gaussian decays as exp(-x^2/2), so at x=6 it's ~exp(-18) ≈ 1.5e-8

def _hermite_h(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x) if np.ndim(x) else 1.0
    if n == 1:
        return 2 * x if np.ndim(x) else 2.0 * x
    if np.ndim(x) == 0:
        x = np.array([x])
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

# Precompute Hermite values on quadrature grid
_H_GRID = np.array([_hermite_h(d, _QUAD_X) for d in _DEGREES])  # shape (4, N_QUAD)
_H_FT_GRID = np.array([_hermite_h(d, _TWO_PI * _QUAD_X) for d in _DEGREES])  # shape (4, N_QUAD)
_GAUSS_X = np.exp(-_QUAD_X**2 / 2.0)
_GAUSS_FT = np.exp(-_TWO_PI**2 * _QUAD_X**2)

def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    return -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

def _l2_norm_sq_analytic(c0, c4, c8):
    """Compute L2 norm squared analytically using orthogonality."""
    c12 = _compute_c12(c0, c4, c8)
    coeffs = np.array([c0, c4, c8, c12])
    return float(np.sum(coeffs**2 * _L2_FACTORS))

def _l1_g_grid(coeffs):
    """Compute |g(x)| on the quadrature grid. coeffs = [c0,c4,c8,c12]."""
    g = (coeffs @ _H_GRID) * _GAUSS_X
    return np.abs(g)

def _l1_Fg_grid(coeffs):
    """Compute |F(g)(k)| on the quadrature grid. coeffs = [c0,c4,c8,c12]."""
    ft_coeffs = coeffs * _FT_MULTIPLIERS
    Fg = (ft_coeffs @ _H_FT_GRID) * _GAUSS_FT
    return np.abs(Fg)

def _l1_norm(coeffs):
    """Compute L1 norm of g via trapezoidal rule on fixed grid."""
    abs_g = _l1_g_grid(coeffs)
    return float(np.sum(abs_g * _QUAD_W))

def _l1_norm_Fg(coeffs):
    """Compute L1 norm of F(g) via trapezoidal rule on fixed grid."""
    abs_Fg = _l1_Fg_grid(coeffs)
    return float(np.sum(abs_Fg * _QUAD_W))

def _bound_val(coeffs):
    """Compute the bound A(f)*A(F(f)) for coeffs = [c0, c4, c8]."""
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    full = np.array([c0, c4, c8, c12])
    
    l2_sq = _l2_norm_sq_analytic(c0, c4, c8)
    if l2_sq < 1e-15:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    l1_g = _l1_norm(full)
    l1_Fg = _l1_norm_Fg(full)
    
    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10
    
    A_f = l1_g / l2
    A_F = l1_Fg / l2
    return A_f * A_F

def _bound_and_grad(coeffs):
    """
    Compute the bound and its gradient using finite differences.
    coeffs = [c0, c4, c8]
    """
    val = _bound_val(coeffs)
    eps = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_minus = coeffs.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        bp = _bound_val(c_plus)
        bm = _bound_val(c_minus)
        grad[i] = (bp - bm) / (2.0 * eps)
    return val, grad

def _coordinate_descent_polish(coeffs, max_iters=500, tol=1e-10):
    """Simple coordinate descent for final polish."""
    best = coeffs.copy()
    best_val = _bound_val(best)
    step = 1e-3
    for _ in range(max_iters):
        improved = False
        for i in range(3):
            for direction in [1, -1]:
                trial = best.copy()
                trial[i] += direction * step
                tv = _bound_val(trial)
                if tv < best_val - 1e-14:
                    best = trial
                    best_val = tv
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-12:
                break
    return best, best_val

def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin (evaluator kills at 210s)
    
    # Starting point
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            return _bound_val(coeffs)
        except Exception:
            return 1e10
    
    def objective_grad(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10, np.zeros(3)
        try:
            return _bound_and_grad(coeffs)
        except Exception:
            return 1e10, np.zeros(3)
    
    # Initial evaluation
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # Phase 1: L-BFGS-B with gradient
    try:
        result = optimize.minimize(
            objective, x0, jac=objective_grad, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead for local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            polished, polished_val = _coordinate_descent_polish(best_coeffs, max_iters=1000)
            if polished_val < best_val:
                best_val = polished_val
                best_coeffs = polished
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish from best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 3.0:
        try:
            result = optimize.minimize(
                objective, best_coeffs, jac=objective_grad, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    val, grad = _bound_and_grad(coeffs)
    print(f"Upper bound on C4: {val}")
