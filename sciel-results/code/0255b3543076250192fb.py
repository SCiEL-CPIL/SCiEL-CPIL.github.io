"""Rewritten program with coordinate descent exploiting rational structure."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 200
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials evaluated at quadrature nodes
_x2 = _x_gh ** 2
_H0 = np.ones_like(_x_gh)
_H4 = 16*_x2**2 - 48*_x2 + 12
_H8 = 256*_x2**4 - 3072*_x2**3 + 11520*_x2**2 - 13440*_x2 + 1680
_H12 = (4096*_x2**6 - 73728*_x2**5 + 580608*_x2**4 - 2419200*_x2**3 
        + 5443200*_x2**2 - 5443200*_x2 + 2032128)

# Derivatives of Hermite polynomials
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Hermite values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 2032128.0

# Pre-compute inner products for the rational function structure
# We need integrals of the form ∫ H_i * H_j * exp(-x^2) dx and ∫ H_i * x * H_j * exp(-x^2) dx
# and similar for derivatives

# For the coordinate descent, we treat the bound as:
# sqrt( (∫x² P² e^{-x²}) / (∫ P² e^{-x²}) * (∫ (P' - xP)² e^{-x²}) / (∫ P² e^{-x²}) )
#
# When varying one coefficient a_k, P = a_k * H_k + Q where Q is fixed.
# Let's pre-compute the necessary quadratic form matrices.

# Build the 4x4 Gram matrix G[i,j] = ∫ H_i * H_j * exp(-x^2) dx
# where H_0=H0, H_1=H4, H_2=H8, H_3=H12
_H_all = np.column_stack([_H0, _H4, _H8, _H12])
_G = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _G[i,j] = np.sum(_w_gh * _H_all[:,i] * _H_all[:,j])

# Second moment matrix: M[i,j] = ∫ x² * H_i * H_j * exp(-x^2) dx
_M = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _M[i,j] = np.sum(_w_gh * _x2 * _H_all[:,i] * _H_all[:,j])

# Derivative vectors
_dH_all = np.column_stack([np.zeros_like(_x_gh), _dH4, _dH8, _dH12])

# Momentum integrals: ∫ (P' - xP)² e^{-x^2} dx
# = ∫ P'² e^{-x^2} - 2∫ P' * x * P * e^{-x^2} + ∫ x² P² e^{-x^2}
# Let's compute:
# D[i,j] = ∫ dH_i * dH_j * exp(-x^2) dx
_D = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _D[i,j] = np.sum(_w_gh * _dH_all[:,i] * _dH_all[:,j])

# E[i,j] = ∫ dH_i * x * H_j * exp(-x^2) dx
_E = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _E[i,j] = np.sum(_w_gh * _dH_all[:,i] * _x_gh * _H_all[:,j])

# F[i,j] = ∫ x * H_i * H_j * exp(-x^2) dx (should be 0 for even functions, but compute anyway)
_F = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _F[i,j] = np.sum(_w_gh * _x_gh * _H_all[:,i] * _H_all[:,j])

def _compute_bound_fast(a0, a4, a8):
    """Compute the uncertainty product using pre-computed matrices."""
    a12 = -(a0 + _H4_0*a4 + _H8_0*a8) / _H12_0
    a = np.array([a0, a4, a8, a12])
    
    norm2 = a @ _G @ a
    if norm2 < 1e-30:
        return 1e10
    
    mom2 = a @ _M @ a
    var_x = mom2 / norm2
    
    # Momentum: ∫ (P' - xP)² e^{-x^2}
    # = a'Da - 2*a'Ex*a + a'Ma
    # where Ex is the matrix with (Ex)_{i,j} = ∫ dH_i * x * H_j * e^{-x²} dx = E[i,j]
    momentum2 = a @ _D @ a - 2.0 * (a @ _E @ a) + mom2
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _objective(a):
    return _compute_bound_fast(a[0], a[1], a[2])

def _coordinate_descent(a0, a4, a8, max_iter=500, tol=1e-12):
    """
    Coordinate descent exploiting the rational structure.
    For each coefficient, the bound is sqrt of a ratio of quadratic forms.
    We minimize by finding the optimal value analytically or via 1D search.
    """
    best_a = np.array([a0, a4, a8])
    best_val = _compute_bound_fast(a0, a4, a8)
    
    for iteration in range(max_iter):
        improved = False
        for coord in range(3):
            # Fix other two, optimize one
            a_fixed = best_a.copy()
            a_test = a_fixed.copy()
            
            # 1D minimization over the fixed coordinate
            # Use golden section or scipy minimize_scalar
            from scipy.optimize import minimize_scalar
            
            def obj_1d(t):
                a_test[coord] = t
                return _compute_bound_fast(a_test[0], a_test[1], a_test[2])
            
            # Initial bracket
            t0 = a_fixed[coord]
            # Try to find a bracket
            dt = 0.1
            t_lo, t_hi = t0 - 1.0, t0 + 1.0
            f_lo, f_hi = obj_1d(t_lo), obj_1d(t_hi)
            f_mid = obj_1d(t0)
            
            # Expand bracket if needed
            for _ in range(20):
                if f_lo > f_mid and f_hi > f_mid:
                    break
                if f_lo < f_mid:
                    t_hi = t_mid = t_lo
                    t_lo = t_lo - dt
                    dt *= 2
                    f_lo = obj_1d(t_lo)
                    f_mid = obj_1d(t_hi)
                else:
                    t_lo = t_mid = t_hi
                    t_hi = t_hi + dt
                    dt *= 2
                    f_hi = obj_1d(t_hi)
                    f_mid = obj_1d(t_lo)
            
            try:
                res = minimize_scalar(obj_1d, bounds=(t_lo, t_hi), method='bounded',
                                     options={'xatol': 1e-14, 'maxiter': 200})
                if res.fun < best_val - 1e-15:
                    best_a[coord] = res.x
                    best_val = res.fun
                    improved = True
            except Exception:
                pass
        
        if not improved or best_val < tol:
            break
    
    return best_a, best_val

def construct():
    start_time = time.time()
    deadline = 210.0
    safety_margin = 10.0
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound_fast(x0[0], x0[1], x0[2])
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [x0]
    for _ in range(10):
        starts.append(x0 + rng.normal(0, 0.03, size=3))
    for _ in range(8):
        starts.append(x0 + rng.normal(0, 0.2, size=3))
    for _ in range(6):
        starts.append(x0 + rng.normal(0, 0.8, size=3))
    for _ in range(5):
        starts.append(rng.uniform(-2, 2, size=3))
    for a0_try in [-1.5, -1.2, -0.8, -0.5]:
        for a4_try in [0.0, 0.02, 0.05, 0.08]:
            starts.append(np.array([a0_try, a4_try, 0.0003]))
    
    # Phase 1: Multi-start with coordinate descent
    for start in starts:
        if time.time() > deadline - safety_margin - 30:
            break
        try:
            # First do a quick Nelder-Mead to get close
            res_nm = minimize(_objective, start, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            # Then refine with coordinate descent
            cd_x, cd_val = _coordinate_descent(res_nm.x[0], res_nm.x[1], res_nm.x[2], max_iter=200)
            
            if cd_val < best_score:
                best_score = cd_val
                best_x = cd_x.copy()
        except Exception:
            pass
    
    # Phase 2: BFGS refinement from best point
    if time.time() < deadline - safety_margin - 15:
        try:
            res_bfgs = minimize(_objective, best_x, method='BFGS',
                               options={'maxiter': 1000, 'gtol': 1e-12, 'disp': False})
            if res_bfgs.fun < best_score:
                best_score = res_bfgs.fun
                best_x = res_bfgs.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - safety_margin - 10:
        try:
            cd_x, cd_val = _coordinate_descent(best_x[0], best_x[1], best_x[2], max_iter=1000, tol=1e-14)
            if cd_val < best_score:
                best_score = cd_val
                best_x = cd_x.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    if time.time() < deadline - safety_margin:
        try:
            res_polish = minimize(_objective, best_x, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
