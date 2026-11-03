"""Rewritten program for the suite task; EVOLVE-BLOCK optimized with Rayleigh-quotient formulation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = _START + 170.0  # safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# Pre-compute constants for analytic evaluation
# Indices in basis: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Norms of Hermite polynomials: <H_n, H_n> = sqrt(pi) * 2^n * n!
norms = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# Diagonal elements of <x^2 H_i, H_i> = (i + 0.5) * <H_i, H_i>
x2_diag = np.array([(n + 0.5) * norm for n, norm in zip(indices, norms)])

# Values of H_n(0) for n in indices to determine c12
H_vals_at_0 = np.array([1.0, 12.0, 1680.0, 665280.0])

# Odd Hermite norms for Q(x) = P' - xP
odd_indices = [1, 3, 5, 7, 9, 11, 13]
odd_norms = np.array([sqrt(pi) * (2**k) * factorial(k) for k in odd_indices])

# Matrix A for norm: N = c^T A c where c = [c0, c4, c8, c12]
# A is diagonal with A[i,i] = norms[i]
A = np.diag(norms)

# Matrix B for x^2: <x^2> num = c^T B c
B = np.diag(x2_diag)

# For p^2, we need d coefficients in terms of c:
# d1 = -0.5 c0
# d3 = 4.0 c4
# d5 = -0.5 c4
# d7 = 8.0 c8
# d9 = -0.5 c8
# d11 = 12.0 c12
# d13 = -0.5 c12
# So p2_num = d^T C d where C = diag(odd_norms)
# Let M be the 4x7 matrix mapping c to d
M = np.zeros((4, 7))
M[0, 0] = -0.5   # c0 -> d1
M[1, 1] = 4.0    # c4 -> d3
M[1, 2] = -0.5   # c4 -> d5
M[2, 3] = 8.0    # c8 -> d7
M[2, 4] = -0.5   # c8 -> d9
M[3, 5] = 12.0   # c12 -> d11
M[3, 6] = -0.5   # c12 -> d13

C = np.diag(odd_norms)

def _compute_bound_analytic(c0, c4, c8):
    """Compute the uncertainty product bound A(f)*A(F) analytically."""
    c12 = -(c0 * H_vals_at_0[0] + c4 * H_vals_at_0[1] + c8 * H_vals_at_0[2]) / H_vals_at_0[3]
    
    coeffs = np.array([c0, c4, c8, c12])
    
    norm_sq = coeffs @ A @ coeffs
    if norm_sq < 1e-15:
        return 1e15
        
    x2_num = coeffs @ B @ coeffs
    delta_x_sq = x2_num / norm_sq
    
    d = M @ coeffs
    p2_num = d @ C @ d
    delta_p_sq = p2_num / norm_sq
    
    return np.sqrt(delta_x_sq * delta_p_sq)

# Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
# c12 = -(c0 + 12*c4 + 1680*c8) / 665280

# Precompute the reduced 3x3 matrices by eliminating c12
# Let c = [c0, c4, c8], and c12 = -K @ c where K = [1, 12, 1680] / 665280
K = np.array([1.0, 12.0, 1680.0]) / 665280.0

# Full 4-vector: c_full = [c0, c4, c8, c12] = [c; -K@c]
# = [I3; -K] @ c
# Let E = [[I3], [-K]] (4x3 matrix)
E = np.zeros((4, 3))
E[:3, :] = np.eye(3)
E[3, :] = -K

# Reduced matrices:
Ar = E.T @ A @ E
Br = E.T @ B @ E
# For p2: d = M @ c_full = M @ E @ c
Me = M @ E  # 7x3
Cr = Me.T @ C @ Me

# Objective is sqrt( (c^T Br c) / (c^T Ar c) * (c^T Cr c) / (c^T Ar c) )
# = sqrt( (c^T Br c) * (c^T Cr c) ) / (c^T Ar c)

def _objective_3d(x):
    """Minimize the squared uncertainty product to avoid sqrt in inner loop."""
    Ar_c = x @ Ar @ x
    if Ar_c < 1e-15:
        return 1e30
    Br_c = x @ Br @ x
    Cr_c = x @ Cr @ x
    return (Br_c * Cr_c) / (Ar_c * Ar_c)

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
_RAW_SEED = _compute_bound_analytic(*_SEED)

def _objective(x):
    return np.sqrt(_objective_3d(x))

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _compute_bound_analytic(*_SEED)
    
    try:
        # 1. Differential Evolution for global search on the 3D problem
        if _time_left() > 50:
            bounds = [(-10, 10), (-10, 10), (-5, 5)]
            res = differential_evolution(
                _objective_3d, bounds, seed=42, maxiter=200, popsize=25,
                tol=1e-10, polish=False, init='sobol'
            )
            if res.x is not None:
                v = _compute_bound_analytic(*res.x)
                if v < best_v:
                    best_v = v
                    best_x = res.x
        
        # 2. Nelder-Mead polish
        if _time_left() > 15:
            for x0 in [best_x, _SEED]:
                try:
                    r2 = minimize(_objective_3d, x0, method='Nelder-Mead',
                                  options={'maxiter':3000, 'xatol':1e-14, 'fatol':1e-16, 'adaptive':True})
                    if r2.fun < 1e29:
                        v = _compute_bound_analytic(*r2.x)
                        if v < best_v:
                            best_v = v
                            best_x = r2.x
                except Exception:
                    pass
        
        # 3. L-BFGS-B final polish
        if _time_left() > 5:
            try:
                r2 = minimize(_objective_3d, best_x, method='L-BFGS-B', options={'maxiter':5000, 'gtol':1e-14})
                if r2.fun < 1e29:
                    v = _compute_bound_analytic(*r2.x)
                    if v < best_v:
                        best_v = v
                        best_x = r2.x
            except Exception:
                pass
        
    except Exception:
        pass
        
    return np.array([best_x[0], best_x[1], best_x[2]])
# EVOLVE-BLOCK-END
