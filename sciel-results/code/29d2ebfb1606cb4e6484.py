"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 200
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials evaluated at quadrature nodes (pre-compute for speed)
_x2 = _x_gh ** 2
_H0 = np.ones_like(_x_gh)
_H4 = 16*_x2**2 - 48*_x2 + 12
_H8 = 256*_x2**4 - 3072*_x2**3 + 11520*_x2**2 - 13440*_x2 + 1680
_H12 = (4096*_x2**6 - 73728*_x2**5 + 580608*_x2**4 - 2419200*_x2**3 
        + 5443200*_x2**2 - 5443200*_x2 + 2032128)

# Pre-compute derivatives of Hermite polynomials at quadrature nodes
# d/dx H4 = 64x^3 - 96x
_dH4 = 64*_x_gh**3 - 96*_x_gh
# d/dx H8 = 2048x^7 - 8064x^5 + 10752x^3 - 1344x
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
# d/dx H12 = 49152x^11 - 322560x^9 + 677376x^7 - 653184x^5 + 258048x^3 - 14336x
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    # Compute a12 from constraint P(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    # Build P(x) at quadrature nodes
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    
    # f(x) = P(x) * exp(-x^2/2)
    # |f|^2 = P^2 * exp(-x^2)
    f_sq = P**2
    
    # ||f||_2^2 = integral P^2 exp(-x^2) dx
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10  # Avoid division by zero
    
    # Second moment: integral x^2 |f|^2 dx
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    
    # Variance in x: <x^2> - <x>^2, but since P(0)=0 and even polynomials, <x>=0
    # Actually P is even (all terms are even powers), so <x> = 0
    var_x = mom2 / norm2
    
    # f'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    # |f'|^2 = (P' - x*P)^2 * exp(-x^2)
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12  # dH0 = 0
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    
    # ||f'||_2^2 = integral |f'|^2 dx
    momentum2 = np.sum(_w_gh * fp_sq)
    
    var_p = momentum2 / norm2
    
    # Uncertainty product
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _objective_grad(a):
    """Compute objective and gradient via finite differences for BFGS."""
    a0, a4, a8 = a
    f_val = _compute_bound(a0, a4, a8)
    
    # Numerical gradient
    eps = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        a_plus = a.copy()
        a_plus[i] += eps
        a_minus = a.copy()
        a_minus[i] -= eps
        grad[i] = (_compute_bound(a_plus[0], a_plus[1], a_plus[2]) - 
                   _compute_bound(a_minus[0], a_minus[1], a_minus[2])) / (2*eps)
    
    return f_val, grad

def construct():
    start_time = time.time()
    deadline = 210.0  # Wall-clock limit
    safety_margin = 10.0  # Safety margin in seconds
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    # Generate multiple starting points for global search
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Local perturbations around seed
    for _ in range(10):
        starts.append(x0 + rng.normal(0, 0.03, size=3))
    # Medium exploration
    for _ in range(8):
        starts.append(x0 + rng.normal(0, 0.2, size=3))
    # Wider exploration
    for _ in range(6):
        starts.append(x0 + rng.normal(0, 0.8, size=3))
    # Random wide starts
    for _ in range(5):
        starts.append(rng.uniform(-2, 2, size=3))
    # Structured starts: try different ratios
    for a0_try in [-1.5, -1.2, -0.8, -0.5]:
        for a4_try in [0.0, 0.02, 0.05, 0.08]:
            starts.append(np.array([a0_try, a4_try, 0.0003]))
    
    def objective(a):
        return _compute_bound(a[0], a[1], a[2])
    
    # Phase 1: Multi-start Nelder-Mead for global exploration
    for start in starts:
        if time.time() > deadline - safety_margin - 30:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 
                                   'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient-based refinement (BFGS) from best point
    if time.time() < deadline - safety_margin - 15:
        try:
            res_bfgs = minimize(objective, best_x, method='BFGS',
                              options={'maxiter': 1000, 'gtol': 1e-12, 
                                       'disp': False})
            if res_bfgs.fun < best_score:
                best_score = res_bfgs.fun
                best_x = res_bfgs.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - safety_margin:
        try:
            res_polish = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 10000, 'xatol': 1e-14, 
                                         'fatol': 1e-16, 'adaptive': True})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more BFGS pass from polished point
    if time.time() < deadline - safety_margin:
        try:
            res_final = minimize(objective, best_x, method='BFGS',
                               options={'maxiter': 500, 'gtol': 1e-14, 
                                        'disp': False})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
