"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 400
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Pre-compute x^2 and Hermite polynomials at quadrature nodes
_x2 = _x_gh ** 2
_x4 = _x2 ** 2
_x6 = _x4 * _x2
_x8 = _x4 ** 2
_x10 = _x8 * _x2
_x12 = _x6 ** 2

_H0 = np.ones_like(_x_gh)
_H4 = 16*_x4 - 48*_x2 + 12
_H8 = 256*_x8 - 3072*_x6 + 11520*_x4 - 13440*_x2 + 1680
_H12 = (4096*_x12 - 73728*_x10 + 580608*_x8 - 2419200*_x6 
        + 5443200*_x4 - 5443200*_x2 + 2032128)

# Pre-compute derivatives of Hermite polynomials at quadrature nodes
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Pre-compute the constraint coefficients
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 2032128.0

def _compute_bound_and_grad(a0, a4, a8):
    """Compute the uncertainty product and its gradient."""
    # Compute a12 from constraint P(0) = 0
    a12 = -(a0 + _H4_0*a4 + _H8_0*a8) / _H12_0
    
    # Build P(x) at quadrature nodes
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    
    # |f|^2 = P^2 * exp(-x^2), so integrand is P^2
    f_sq = P**2
    
    # ||f||_2^2 = integral P^2 exp(-x^2) dx
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10, np.zeros(3)
    
    # Second moment: integral x^2 |f|^2 dx
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    
    # Variance in x (P is even, so <x> = 0)
    var_x = mom2 / norm2
    
    # f'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    
    # ||f'||_2^2 = integral |f'|^2 dx
    momentum2 = np.sum(_w_gh * fp_sq)
    
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10, np.zeros(3)
    
    # Uncertainty product
    bound = np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))
    
    # Compute gradient using finite differences for simplicity
    # (analytic gradient is complex; finite differences are fast enough)
    eps = 1e-7
    grad = np.zeros(3)
    
    # Gradient w.r.t. a0
    val_p = _compute_bound_and_grad(a0 + eps, a4, a8)[0]
    val_m = _compute_bound_and_grad(a0 - eps, a4, a8)[0]
    grad[0] = (val_p - val_m) / (2*eps)
    
    # Gradient w.r.t. a4
    val_p = _compute_bound_and_grad(a0, a4 + eps, a8)[0]
    val_m = _compute_bound_and_grad(a0, a4 - eps, a8)[0]
    grad[1] = (val_p - val_m) / (2*eps)
    
    # Gradient w.r.t. a8
    val_p = _compute_bound_and_grad(a0, a4, a8 + eps)[0]
    val_m = _compute_bound_and_grad(a0, a4, a8 - eps)[0]
    grad[2] = (val_p - val_m) / (2*eps)
    
    return bound, grad

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    return _compute_bound_and_grad(a0, a4, a8)[0]

def construct():
    start_time = time.time()
    deadline = 210.0  # Wall-clock limit
    safety_margin = 15.0  # Safety margin in seconds
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    # Generate diverse starting points for global search
    rng = np.random.default_rng(42)
    
    starts = [x0]
    
    # Local perturbations around seed (fine-tuning)
    for scale in [0.01, 0.05, 0.1]:
        for _ in range(4):
            starts.append(x0 + rng.normal(0, scale, size=3))
    
    # Medium exploration
    for _ in range(6):
        starts.append(x0 + rng.normal(0, 0.5, size=3))
    
    # Wide exploration
    for _ in range(8):
        starts.append(rng.uniform(-5, 5, size=3))
    
    # Targeted exploration in promising regions
    for _ in range(6):
        a0 = rng.uniform(-3, 0)
        a4 = rng.uniform(-0.5, 0.5)
        a8 = rng.uniform(-0.5, 0.5)
        starts.append(np.array([a0, a4, a8]))
    
    def objective(a):
        return _compute_bound(a[0], a[1], a[2])
    
    def objective_grad(a):
        val, grad = _compute_bound_and_grad(a[0], a[1], a[2])
        return val, grad
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for start in starts:
        if time.time() > deadline - safety_margin:
            break
        
        try:
            # First try L-BFGS-B with gradients
            res = minimize(objective, start, method='L-BFGS-B',
                          jac=objective_grad,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - safety_margin:
            break
    
    # Phase 2: Refine best solution with Nelder-Mead (no gradient needed)
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 
                                   'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with L-BFGS-B
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                          jac=objective_grad,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
