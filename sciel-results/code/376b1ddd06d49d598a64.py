"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import differential_evolution, minimize
import time

# Precompute Gauss-Hermite nodes and weights
# N=20 is sufficient for high accuracy
N_QUAD = 20
_x_nodes, _w_weights = hermgauss(N_QUAD)

# Precompute Hermite polynomial values at nodes
_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

# Values at x=0 for constraint (Physicist's Hermite)
_h0_0 = 1.0
_h4_0 = 3.0
_h8_0 = 105.0
_h12_0 = 945.0

# Basis functions for P(x) = c0*A + c4*B + c8*C
# P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# c12 = -(c0 + 3*c4 + 105*c8) / 945
_A = _H0 - _H12 / _h12_0
_B = _H4 - _h4_0 * _H12 / _h12_0
_C = _H8 - _h8_0 * _H12 / _h12_0

_x_sq = _x_nodes**2

def _compute_bound_2d(c4, c8):
    """Compute the upper bound C4 for given c4, c8 with c0=1."""
    c0 = 1.0
    # Compute P(x) at nodes
    P = c0 * _A + c4 * _B + c8 * _C
    
    P_sq = P * P
    norm_sq = np.sum(_w_weights * P_sq)
    
    if norm_sq < 1e-15:
        return 1e10
        
    second_moment = np.sum(_w_weights * _x_sq * P_sq) / norm_sq
    C4 = second_moment ** 2
    return C4

def _compute_bound_3d(c0, c4, c8):
    """Compute the upper bound C4 for given coefficients."""
    P = c0 * _A + c4 * _B + c8 * _C
    P_sq = P * P
    norm_sq = np.sum(_w_weights * P_sq)
    if norm_sq < 1e-15:
        return 1e10
    second_moment = np.sum(_w_weights * _x_sq * P_sq) / norm_sq
    C4 = second_moment ** 2
    return C4

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin

    best_c = None
    best_val = float('inf')

    # Strategy 1: 2D optimization with c0=1
    try:
        bounds_2d = [(-10, 10), (-10, 10)]
        res = differential_evolution(
            _compute_bound_2d, 
            bounds_2d, 
            maxiter=300, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            polish=False,
            init='sobol'
        )
        if res.fun < best_val:
            best_val = res.fun
            best_c = np.array([1.0, res.x[0], res.x[1]])
    except Exception:
        pass

    # Polish the 2D result
    if best_c is not None:
        try:
            res_local = minimize(
                _compute_bound_2d, 
                best_c[1:], 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_c = np.array([1.0, res_local.x[0], res_local.x[1]])
        except Exception:
            pass
        try:
            res_nm = minimize(
                _compute_bound_2d, 
                best_c[1:], 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_c = np.array([1.0, res_nm.x[0], res_nm.x[1]])
        except Exception:
            pass

    # Strategy 2: 3D optimization to catch cases where c0 != 1 is better
    # (though scale invariance suggests c0=1 is WLOG, numerical issues might arise)
    # Or just to be safe, run a quick 3D DE
    if time.time() < deadline - 20:
        try:
            bounds_3d = [(-10, 10), (-10, 10), (-10, 10)]
            res3 = differential_evolution(
                _compute_bound_3d, 
                bounds_3d, 
                maxiter=200, 
                popsize=15, 
                tol=1e-10, 
                seed=123,
                polish=False,
                init='sobol'
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_c = res3.x.copy()
                
                # Polish 3D result
                res3_local = minimize(
                    _compute_bound_3d, 
                    best_c, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res3_local.fun < best_val:
                    best_val = res3_local.fun
                    best_c = res3_local.x.copy()
        except Exception:
            pass

    if best_c is None:
        # Fallback to seed-like values
        return np.array([-1.13, 0.04, 0.00030864])
        
    return np.array(best_c)

# EVOLVE-BLOCK-END
