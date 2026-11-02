"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 300
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
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Pre-compute x * H terms for efficiency
_xH4 = _x_gh * _H4
_xH8 = _x_gh * _H8
_xH12 = _x_gh * _H12

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
    var_x = mom2 / norm2
    
    # f'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    # |f'|^2 = (P' - x*P)^2 * exp(-x^2)
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12  # dH0 = 0
    xP = a0*_x_gh*_H0 + a4*_xH4 + a8*_xH8 + a12*_xH12
    fp_inner = dP - xP
    fp_sq = fp_inner**2
    
    # ||f'||_2^2 = integral |f'|^2 dx
    momentum2 = np.sum(_w_gh * fp_sq)
    
    var_p = momentum2 / norm2
    
    # Uncertainty product
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _compute_bound_fast(a):
    """Vectorized version for speed."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    f_sq = P**2
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10
    
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    var_x = mom2 / norm2
    
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    xP = a0*_x_gh*_H0 + a4*_xH4 + a8*_xH8 + a12*_xH12
    fp_inner = dP - xP
    fp_sq = fp_inner**2
    momentum2 = np.sum(_w_gh * fp_sq)
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def construct():
    start_time = time.time()
    deadline = 200.0  # Wall-clock limit with safety margin
    safety_margin = 10.0
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    # Generate multiple starting points for global search
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Local perturbations around seed
    for _ in range(15):
        starts.append(x0 + rng.normal(0, 0.02, size=3))
    # Medium exploration
    for _ in range(12):
        starts.append(x0 + rng.normal(0, 0.15, size=3))
    # Wider exploration
    for _ in range(10):
        starts.append(x0 + rng.normal(0, 0.5, size=3))
    # Random wide starts
    for _ in range(8):
        starts.append(rng.uniform(-3, 3, size=3))
    # Structured starts: try different ratios
    for a0_try in [-2.0, -1.5, -1.2, -0.9, -0.6, -0.3]:
        for a4_try in [-0.05, 0.0, 0.02, 0.05, 0.08, 0.12]:
            starts.append(np.array([a0_try, a4_try, 0.0002]))
    # More diverse structured starts
    for a0_try in [-1.5, -1.0, -0.5]:
        for a8_try in [0.0001, 0.0005, 0.001, 0.002]:
            starts.append(np.array([a0_try, 0.03, a8_try]))
    
    def objective(a):
        return _compute_bound(a[0], a[1], a[2])
    
    # Phase 1: Multi-start Nelder-Mead for global exploration
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin - 40:
            break
        
        try:
            # Use fewer iterations for early starts, more for promising ones
            max_iter = 3000 if i < 10 else 1500
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 
                                   'fatol': 1e-14, 'adaptive': True,
                                   'initial_simplex': None})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient-based refinement (L-BFGS-B) from best point
    if time.time() - start_time < deadline - safety_margin - 20:
        try:
            res_bfgs = minimize(objective, best_x, method='L-BFGS-B',
                              options={'maxiter': 2000, 'gtol': 1e-14, 
                                       'disp': False, 'ftol': 1e-16})
            if res_bfgs.fun < best_score:
                best_score = res_bfgs.fun
                best_x = res_bfgs.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coarse-to-fine local search
    if time.time() - start_time < deadline - safety_margin - 10:
        # Try multiple local searches with decreasing step sizes
        for scale in [0.1, 0.05, 0.01, 0.001]:
            if time.time() - start_time > deadline - safety_margin - 5:
                break
            try:
                res_local = minimize(objective, best_x, method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-15, 
                                             'fatol': 1e-18, 'adaptive': True,
                                             'initial_simplex': np.array([
                                                 best_x + scale*np.array([1,0,0]),
                                                 best_x + scale*np.array([0,1,0]),
                                                 best_x + scale*np.array([0,0,1]),
                                                 best_x])})
                if res_local.fun < best_score:
                    best_score = res_local.fun
                    best_x = res_local.x.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() - start_time < deadline - safety_margin:
        try:
            res_polish = minimize(objective, best_x, method='Nelder-Mead',
                                options={'maxiter': 20000, 'xatol': 1e-16, 
                                         'fatol': 1e-18, 'adaptive': True})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 5: One more BFGS pass from polished point
    if time.time() - start_time < deadline - safety_margin:
        try:
            res_final = minimize(objective, best_x, method='L-BFGS-B',
                               options={'maxiter': 1000, 'gtol': 1e-16, 
                                        'disp': False, 'ftol': 1e-18})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    # Phase 6: Very fine local search
    if time.time() - start_time < deadline - safety_margin:
        try:
            res_fine = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 50000, 'xatol': 1e-18, 
                                        'fatol': 1e-20, 'adaptive': True,
                                        'initial_simplex': np.array([
                                            best_x + 1e-6*np.array([1,0,0]),
                                            best_x + 1e-6*np.array([0,1,0]),
                                            best_x + 1e-6*np.array([0,0,1]),
                                            best_x])})
            if res_fine.fun < best_score:
                best_score = res_fine.fun
                best_x = res_fine.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
