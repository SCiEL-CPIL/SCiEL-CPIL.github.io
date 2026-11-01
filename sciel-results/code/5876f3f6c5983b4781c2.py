"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import factorial, sqrt

# Precompute weights w_k = 2^k k! sqrt(pi) for k in {0, 4, 8, 12}
# and potentially for b_m which involves m up to 13 (from 12+1) or 11 (from 12-1)
# b_m is non-zero for m in {1, 3, 5, 7, 9, 11, 13}
# m=1: from c4 (k=4, m=k-1=3? No. b_m = (m+1)c_{m+1} - 0.5 c_{m-1})
# Let's list non-zero c_k: 0, 4, 8, 12.
# b_m non-zero if m+1 in {0,4,8,12} or m-1 in {0,4,8,12}.
# m+1 in {0,4,8,12} => m in {-1, 3, 7, 11} => m in {3, 7, 11}
# m-1 in {0,4,8,12} => m in {1, 5, 9, 13}
# So m in {1, 3, 5, 7, 9, 11, 13}.

_KS = [0, 4, 8, 12]
_WS = [2**k * factorial(k) * sqrt(np.pi) for k in _KS]
# w_0 = sqrt(pi)
# w_4 = 16 * 24 * sqrt(pi) = 384 sqrt(pi)
# w_8 = 256 * 40320 * sqrt(pi) = 10321920 sqrt(pi)
# w_12 = 4096 * 479001600 * sqrt(pi) = 1962026624000 sqrt(pi)

_MS = [1, 3, 5, 7, 9, 11, 13]
_WM = [2**m * factorial(m) * sqrt(np.pi) for m in _MS]

# Indices for c_k in _KS
_IDX_0 = 0
_IDX_4 = 1
_IDX_8 = 2
_IDX_12 = 3

# Coefficients for c12 = -(c0 + 12 c4 + 16800 c8) / 47900160
_CONST_0 = 1.0
_CONST_4 = 12.0
_CONST_8 = 16800.0
_CONST_12 = 47900160.0

def _uncertainty_product_analytic(c0, c4, c8):
    """
    Compute the uncertainty product A(f) * A(F̂) analytically.
    """
    c12 = -(_CONST_0 * c0 + _CONST_4 * c4 + _CONST_8 * c8) / _CONST_12
    c = [c0, c4, c8, c12]
    
    # N = sum c_k^2 w_k
    N = sum(c[i]**2 * _WS[i] for i in range(4))
    if N <= 0 or not np.isfinite(N):
        return np.inf
        
    # X2 = sum c_k^2 (k + 0.5) w_k
    X2 = sum(c[i]**2 * (_KS[i] + 0.5) * _WS[i] for i in range(4))
    
    # b_m = (m+1) c_{m+1} - 0.5 c_{m-1}
    # c_{-1} = 0, c_{13} = 0 (since only 0,4,8,12 exist)
    # We need c_{m+1} and c_{m-1} for m in _MS
    # Map m to index in _KS if m+1 or m-1 is in _KS
    # Helper to get c_j
    def get_c(j):
        if j in _KS:
            return c[_KS.index(j)]
        return 0.0
        
    P2 = 0.0
    for i, m in enumerate(_MS):
        bm = (m + 1) * get_c(m + 1) - 0.5 * get_c(m - 1)
        P2 += bm**2 * _WM[i]
        
    if P2 <= 0 or not np.isfinite(P2):
        return np.inf
        
    A_f_sq = X2 / N
    A_Fhat_sq = P2 / N
    
    if A_f_sq < 0 or A_Fhat_sq < 0:
        return np.inf
        
    return sqrt(A_f_sq * A_Fhat_sq)

def _objective(c):
    return _uncertainty_product_analytic(c[0], c[1], c[2])

def _objective_grad(c):
    """
    Compute gradient of the objective function.
    """
    c0, c4, c8 = c
    c12 = -(_CONST_0 * c0 + _CONST_4 * c4 + _CONST_8 * c8) / _CONST_12
    c = [c0, c4, c8, c12]
    
    # Derivatives of c_k with respect to c0, c4, c8
    # d c0 / d c0 = 1, d c0 / d c4 = 0, d c0 / d c8 = 0
    # d c4 / d c0 = 0, d c4 / d c4 = 1, d c4 / d c8 = 0
    # d c8 / d c0 = 0, d c8 / d c4 = 0, d c8 / d c8 = 1
    # d c12 / d c0 = -1/47900160, d c12 / d c4 = -12/47900160, d c12 / d c8 = -16800/47900160
    
    dc = np.zeros((4, 3))
    dc[0, 0] = 1.0
    dc[1, 1] = 1.0
    dc[2, 2] = 1.0
    dc[3, 0] = -_CONST_0 / _CONST_12
    dc[3, 1] = -_CONST_4 / _CONST_12
    dc[3, 2] = -_CONST_8 / _CONST_12
    
    N = sum(c[i]**2 * _WS[i] for i in range(4))
    X2 = sum(c[i]**2 * (_KS[i] + 0.5) * _WS[i] for i in range(4))
    
    # dN/dc_i = sum 2 c_k w_k dc_k/dc_i
    dN = np.zeros(3)
    for i in range(3):
        for k in range(4):
            dN[i] += 2 * c[k] * _WS[k] * dc[k, i]
            
    # dX2/dc_i = sum 2 c_k (k+0.5) w_k dc_k/dc_i
    dX2 = np.zeros(3)
    for i in range(3):
        for k in range(4):
            dX2[i] += 2 * c[k] * (_KS[k] + 0.5) * _WS[k] * dc[k, i]
            
    # P2 = sum b_m^2 w_m
    # b_m = (m+1) c_{m+1} - 0.5 c_{m-1}
    # db_m/dc_i = (m+1) dc_{m+1}/dc_i - 0.5 dc_{m-1}/dc_i
    P2 = 0.0
    dBm = np.zeros((len(_MS), 3))
    for i, m in enumerate(_MS):
        # Get c_{m+1} and c_{m-1}
        # We need their derivatives
        # dc_{j}/dc_i is 0 unless j is 0,4,8,12
        # If j is in _KS, dc_{j}/dc_i = dc[j_idx, i]
        
        j_plus = m + 1
        j_minus = m - 1
        
        dc_plus = np.zeros(3)
        dc_minus = np.zeros(3)
        
        if j_plus in _KS:
            idx_plus = _KS.index(j_plus)
            dc_plus = dc[idx_plus, :]
        if j_minus in _KS:
            idx_minus = _KS.index(j_minus)
            dc_minus = dc[idx_minus, :]
            
        dBm[i, :] = (m + 1) * dc_plus - 0.5 * dc_minus
        
        # b_m value
        c_plus = 0.0
        c_minus = 0.0
        if j_plus in _KS:
            c_plus = c[_KS.index(j_plus)]
        if j_minus in _KS:
            c_minus = c[_KS.index(j_minus)]
        bm = (m + 1) * c_plus - 0.5 * c_minus
        P2 += bm**2 * _WM[i]
        
    dP2 = np.zeros(3)
    for i in range(3):
        for m_idx in range(len(_MS)):
            # b_m value again (or store it)
            j_plus = _MS[m_idx] + 1
            j_minus = _MS[m_idx] - 1
            c_plus = 0.0
            c_minus = 0.0
            if j_plus in _KS:
                c_plus = c[_KS.index(j_plus)]
            if j_minus in _KS:
                c_minus = c[_KS.index(j_minus)]
            bm = (_MS[m_idx] + 1) * c_plus - 0.5 * c_minus
            dP2[i] += 2 * bm * _WM[m_idx] * dBm[m_idx, i]
            
    # J = sqrt(X2 * P2) / N
    # dJ/dc_i = (1/N) * 0.5 * (X2 * P2)^-0.5 * (dX2 * P2 + X2 * dP2) - (1/N^2) * sqrt(X2 * P2) * dN
    # Let S = sqrt(X2 * P2)
    # dJ/dc_i = (1/(2 N S)) (dX2 * P2 + X2 * dP2) - S/N^2 * dN
    
    S = sqrt(X2 * P2)
    grad = np.zeros(3)
    for i in range(3):
        grad[i] = (1.0 / (2.0 * N * S)) * (dX2[i] * P2 + X2 * dP2[i]) - (S / (N * N)) * dN[i]
        
    return grad

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200.0
    
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_value = _objective(best_coeffs)
    
    if not np.isfinite(best_value):
        best_value = 1e10
        
    # Multi-start L-BFGS-B with analytic gradient
    initial_points = [
        best_coeffs,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.1, 0.05, 0.0005]),
        np.array([-1.05, 0.08, 0.0008]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.2, 0.02, 0.0003]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-2.0, 0.2, 0.005]),
    ]
    
    for init in initial_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective,
                init,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
            
    return best_coeffs

# EVOLVE-BLOCK-END
