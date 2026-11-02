"""Refined optimization for the Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 300
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials evaluated at quadrature nodes
_x2 = _x_gh ** 2
_H0 = np.ones_like(_x_gh)
_H4 = 16*_x2**2 - 48*_x2 + 12
_H8 = 256*_x2**4 - 3072*_x2**3 + 11520*_x2**2 - 13440*_x2 + 1680
_H12 = (4096*_x2**6 - 73728*_x2**5 + 580608*_x2**4 - 2419200*_x2**3 
        + 5443200*_x2**2 - 5443200*_x_gh**2 + 2032128)

# Pre-compute derivatives of Hermite polynomials at quadrature nodes
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Pre-compute second derivatives for potential use
_ddH4 = 192*_x_gh**2 - 96
_ddH8 = 14336*_x_gh**6 - 40320*_x_gh**4 + 32256*_x_gh**2 - 1344
_ddH12 = (540672*_x_gh**10 - 2903040*_x_gh**8 + 4741632*_x_gh**6 
           - 3265920*_x_gh**4 + 774144*_x_gh**2 - 14336)

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    f_sq = P**2
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10
    
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    var_x = mom2 / norm2
    
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    momentum2 = np.sum(_w_gh * fp_sq)
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return math.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _compute_bound_gradient(a0, a4, a8):
    """Compute the gradient of the bound with respect to (a0, a4, a8)."""
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    f_sq = P**2
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    var_x = mom2 / norm2
    
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    momentum2 = np.sum(_w_gh * fp_sq)
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return np.array([0.0, 0.0, 0.0])
    
    bound_val = math.sqrt(max(var_x, 1e-30) * max(var_p, 1e-30))
    if bound_val < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    dP_da0 = _H0 - (1.0/2032128.0)*_H12
    dP_da4 = _H4 - (12.0/2032128.0)*_H12
    dP_da8 = _H8 - (1680.0/2032128.0)*_H12
    
    ddP_da0 = -(1.0/2032128.0)*_dH12
    ddP_da4 = _dH4 - (12.0/2032128.0)*_dH12
    ddP_da8 = _dH8 - (1680.0/2032128.0)*_dH12
    
    dnorm2_da0 = 2.0 * np.sum(_w_gh * P * dP_da0)
    dnorm2_da4 = 2.0 * np.sum(_w_gh * P * dP_da4)
    dnorm2_da8 = 2.0 * np.sum(_w_gh * P * dP_da8)
    
    dmom2_da0 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da0)
    dmom2_da4 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da4)
    dmom2_da8 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da8)
    
    dfp_da0 = ddP_da0 - _x_gh * dP_da0
    dfp_da4 = ddP_da4 - _x_gh * dP_da4
    dfp_da8 = ddP_da8 - _x_gh * dP_da8
    
    dmom2p_da0 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da0)
    dmom2p_da4 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da4)
    dmom2p_da8 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da8)
    
    dvar_x_da0 = (dmom2_da0 * norm2 - mom2 * dnorm2_da0) / (norm2 * norm2)
    dvar_x_da4 = (dmom2_da4 * norm2 - mom2 * dnorm2_da4) / (norm2 * norm2)
    dvar_x_da8 = (dmom2_da8 * norm2 - mom2 * dnorm2_da8) / (norm2 * norm2)
    
    dvar_p_da0 = (dmom2p_da0 * norm2 - momentum2 * dnorm2_da0) / (norm2 * norm2)
    dvar_p_da4 = (dmom2p_da4 * norm2 - momentum2 * dnorm2_da4) / (norm2 * norm2)
    dvar_p_da8 = (dmom2p_da8 * norm2 - momentum2 * dnorm2_da8) / (norm2 * norm2)
    
    grad0 = (var_p * dvar_x_da0 + var_x * dvar_p_da0) / (2.0 * bound_val)
    grad4 = (var_p * dvar_x_da4 + var_x * dvar_p_da4) / (2.0 * bound_val)
    grad8 = (var_p * dvar_x_da8 + var_x * dvar_p_da8) / (2.0 * bound_val)
    
    return np.array([grad0, grad4, grad8])

def construct():
    start_time = time.time()
    deadline = 195.0  # Wall-clock limit with safety margin before 210s kill
    
    def elapsed():
        return time.time() - start_time
    
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    def objective(a):
        return _compute_bound(a[0], a12_from_a(a), a[2])
    
    def a12_from_a(a):
        return -(a[0] + 12.0*a[1] + 1680.0*a[2]) / 2032128.0
    
    def gradient(a):
        return _compute_bound_gradient(a[0], a[1], a[2])
    
    rng = np.random.default_rng(42)
    
    # Build diverse starting points
    starts = [x0.copy()]
    
    # Local perturbations with multiple scales
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for _ in range(10):
            starts.append(x0 + rng.normal(0, scale, size=3))
    
    # Random wide starts
    for _ in range(30):
        starts.append(rng.uniform(-5, 5, size=3))
    
    # Directional exploration along principal axes
    for axis in range(3):
        for delta in [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]:
            pt = x0.copy()
            pt[axis] += delta
            starts.append(pt)
    
    # Scale-invariant directions
    for _ in range(15):
        scale = 10 ** rng.uniform(-2, 2)
        sign = rng.choice([-1, 1])
        starts.append(x0 * sign * scale)
    
    # Log-uniform exploration
    for _ in range(20):
        pt = np.zeros(3)
        for i in range(3):
            sign = rng.choice([-1, 1])
            mag = 10 ** rng.uniform(-4, 1)
            pt[i] = sign * mag
        starts.append(pt)
    
    # Mix of scales
    for _ in range(20):
        pt = x0 * rng.uniform(0.1, 10.0) + rng.normal(0, 0.5, size=3)
        starts.append(pt)
    
    # Sort starts by initial objective value to prioritize promising ones
    starts_with_scores = [(s, _compute_bound(s[0], s[1], s[2])) for s in starts]
    starts_with_scores.sort(key=lambda x: x[1])
    starts = [s for s, _ in starts_with_scores]
    
    # Multi-start optimization with gradient information
    n_starts = len(starts)
    for i, start in enumerate(starts):
        if elapsed() > deadline - 30.0:
            break
        
        try:
            # Quick pass
            res_quick = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if res_quick.fun < best_score:
                best_score = res_quick.fun
                best_x = res_quick.x.copy()
            
            # Deeper optimization if promising
            if res_quick.fun < best_score * 1.1 and elapsed() < deadline - 25.0:
                res = minimize(objective, res_quick.x, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 5000, 'gtol': 1e-13, 'ftol': 1e-15})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x.copy()
                    
        except Exception:
            pass
    
    # High-precision polish pass
    if elapsed() < deadline - 20.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 20000, 'gtol': 1e-15, 'ftol': 1e-17})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead for non-smooth check
    if elapsed() < deadline - 15.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-18,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish
    if elapsed() < deadline - 10.0 and best_score > 0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-18})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Differential evolution for global exploration if time permits
    if elapsed() < deadline - 40.0:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(objective, bounds, seed=42, 
                                               maxiter=100, tol=1e-12,
                                               popsize=15, init='sobol')
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_x = de_result.x.copy()
                
                # Polish the DE result
                if elapsed() < deadline - 20.0:
                    res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': 10000, 'gtol': 1e-15, 'ftol': 1e-17})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
