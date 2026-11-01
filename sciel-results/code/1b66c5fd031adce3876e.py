"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Structural change: Use Gauss-Hermite quadrature matched to the Gaussian weight,
and derive exact analytic gradients for the L1 norms. This eliminates finite-difference
overhead and improves quadrature accuracy for the Gaussian-weighted integrands.
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

# Use Gauss-Hermite quadrature for L1 norms
# The integrand is |sum c_i H_i(x) exp(-x^2/2)|, which has Gaussian weight exp(-x^2/2)
# We can rewrite: exp(-x^2/2) = exp(-(x/sqrt(2))^2) * exp(0) ... 
# Actually, let's use a change of variable or just use high-order Gauss-Hermite
# For physicists' Hermite: ∫ f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
# Our weight is exp(-x^2/2), so let y = x/sqrt(2), dx = sqrt(2) dy
# ∫ f(x) exp(-x^2/2) dx = sqrt(2) ∫ f(sqrt(2)*y) exp(-y^2) dy
# So we use Gauss-Hermite with the transformed function

_GH_N = 80  # Number of Gauss-Hermite points
_gh_x, _gh_w = np.polynomial.hermite.hermgauss(_GH_N)  # Nodes and weights for exp(-x^2)
# Transform: x = sqrt(2) * y, so integral = sqrt(2) * sum w_i f(sqrt(2)*y_i)
_QUAD_X = np.sqrt(2.0) * _gh_x  # Actual x values
_QUAD_W = np.sqrt(2.0) * _gh_w  # Transformed weights

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

def _compute_bound_and_exact_grad(coeffs):
    """
    Compute the bound and its EXACT gradient.
    coeffs = [c0, c4, c8]
    
    The bound is B = L1(g) * L1(F(g)) / L2^2
    
    Exact gradient:
    dB/dc_j = [dL1(g)/dc_j * L1(F(g)) + L1(g) * dL1(F(g))/dc_j] / L2^2 
              - 2 * B * c_j * L2_factor_j / L2^2
    
    where:
    dL1(g)/dc_j = ∫ sign(g(x)) * H_j(x) * exp(-x^2/2) dx
    dL1(F(g))/dc_j = ∫ sign(F(g)(k)) * FT_mult_j * H_j(2πk) * exp(-(2πk)^2) dk
    """
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    full = np.array([c0, c4, c8, c12])
    
    l2_sq = _l2_norm_sq_analytic(c0, c4, c8)
    if l2_sq < 1e-15:
        return 1e10, np.zeros(3)
    l2 = np.sqrt(l2_sq)
    
    # Compute g(x) on grid
    g_vals = (full @ _H_GRID) * _GAUSS_X  # shape (N_QUAD,)
    abs_g = np.abs(g_vals)
    l1_g = float(np.sum(abs_g * _QUAD_W))
    
    # Compute F(g)(k) on grid
    ft_coeffs = full * _FT_MULTIPLIERS
    Fg_vals = (ft_coeffs @ _H_FT_GRID) * _GAUSS_FT  # shape (N_QUAD,)
    abs_Fg = np.abs(Fg_vals)
    l1_Fg = float(np.sum(abs_Fg * _QUAD_W))
    
    if l1_g < 1e-15 or l1_Fg < 1e-15:
        return 1e10, np.zeros(3)
    
    B = (l1_g / l2) * (l1_Fg / l2)
    
    # Compute exact gradient
    # sign(g(x)) and sign(F(g)(k))
    sign_g = np.sign(g_vals)
    sign_Fg = np.sign(Fg_vals)
    
    # dL1(g)/dc_j = ∫ sign(g(x)) * H_j(x) * exp(-x^2/2) dx
    # For j in {0, 4, 8, 12}, but we only need gradient w.r.t. c0, c4, c8
    # Note: c12 depends on c0, c4, c8, so we need chain rule
    # dL1(g)/dc_j = ∫ sign(g(x)) * [H_j(x) + dc12/dc_j * H_12(x)] * exp(-x^2/2) dx
    # dc12/dc0 = -H0_0/H12_0, dc12/dc4 = -H4_0/H12_0, dc12/dc8 = -H8_0/H12_0
    
    dc12_dc0 = -_H0_0 / _H12_0
    dc12_dc4 = -_H4_0 / _H12_0
    dc12_dc8 = -_H8_0 / _H12_0
    
    # Precompute weighted Hermite values for gradient
    # w_x = exp(-x^2/2) * _QUAD_W (already included in _GAUSS_X * _QUAD_W effectively)
    # Actually _GAUSS_X = exp(-x^2/2), and we multiply by _QUAD_W
    weight_x = _GAUSS_X * _QUAD_W  # shape (N_QUAD,)
    weight_k = _GAUSS_FT * _QUAD_W  # shape (N_QUAD,)
    
    # dL1(g)/dc_j for j = 0, 4, 8
    # = ∫ sign(g) * (H_j + dc12/dc_j * H_12) * exp(-x^2/2) dx
    dL1g_dc = np.zeros(3)
    for j, idx in enumerate([0, 1, 2]):  # degrees 0, 4, 8
        dc12_dcj = [dc12_dc0, dc12_dc4, dc12_dc8][j]
        integrand = sign_g * (_H_GRID[idx] + dc12_dcj * _H_GRID[3]) * weight_x
        dL1g_dc[j] = float(np.sum(integrand))
    
    # dL1(F(g))/dc_j for j = 0, 4, 8
    # F(g)(k) = sum_i c_i * FT_mult_i * H_i(2πk) * exp(-(2πk)^2)
    # dF(g)/dc_j = FT_mult_j * H_j(2πk) * exp(-(2πk)^2) + dc12/dc_j * FT_mult_12 * H_12(2πk) * exp(-(2πk)^2)
    dL1Fg_dc = np.zeros(3)
    for j, idx in enumerate([0, 1, 2]):  # degrees 0, 4, 8
        dc12_dcj = [dc12_dc0, dc12_dc4, dc12_dc8][j]
        integrand = sign_Fg * (_FT_MULTIPLIERS[idx] * _H_FT_GRID[idx] + dc12_dcj * _FT_MULTIPLIERS[3] * _H_FT_GRID[3]) * weight_k
        dL1Fg_dc[j] = float(np.sum(integrand))
    
    # Now compute dB/dc_j
    # dB/dc_j = [dL1g_dc[j] * l1_Fg + l1_g * dL1Fg_dc[j]] / l2_sq - 2 * B * full[j] * _L2_FACTORS[j] / l2_sq
    # Wait, the L2 norm includes c12, so:
    # L2^2 = c0^2 * F0 + c4^2 * F4 + c8^2 * F8 + c12^2 * F12
    # d(L2^2)/dc_j = 2*c_j*F_j + 2*c12*F12*dc12/dc_j
    # d(L2)/dc_j = d(L2^2)/dc_j / (2*L2)
    # dB/dc_j = [dL1g_dc[j] * l1_Fg + l1_g * dL1Fg_dc[j]] / l2_sq - B * d(L2^2)/dc_j / l2_sq
    
    grad = np.zeros(3)
    for j, idx in enumerate([0, 1, 2]):
        dc12_dcj = [dc12_dc0, dc12_dc4, dc12_dc8][j]
        dL2sq_dcj = 2.0 * full[idx] * _L2_FACTORS[idx] + 2.0 * full[3] * _L2_FACTORS[3] * dc12_dcj
        grad[j] = (dL1g_dc[j] * l1_Fg + l1_g * dL1Fg_dc[j]) / l2_sq - B * dL2sq_dcj / l2_sq
    
    return B, grad

def _bound_val(coeffs):
    """Compute the bound A(f)*A(F(f)) for coeffs = [c0, c4, c8]."""
    val, _ = _compute_bound_and_exact_grad(coeffs)
    return val

def _bound_and_grad(coeffs):
    """Compute the bound and its exact gradient."""
    return _compute_bound_and_exact_grad(coeffs)

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
    
    # Phase 1: L-BFGS-B with exact gradient
    try:
        result = optimize.minimize(
            objective, x0, jac=objective_grad, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 3000}
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
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
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
            polished, polished_val = _coordinate_descent_polish(best_coeffs, max_iters=2000)
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
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 2000}
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
    print(f"Gradient: {grad}")
