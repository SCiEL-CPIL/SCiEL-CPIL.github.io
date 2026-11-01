"""Corrected uncertainty bound using Gauss-Hermite quadrature for moment matrices."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss, hermval

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # 15s safety margin before 210s kill

# Precompute quadrature nodes and weights
# N=50 is sufficient for exact integration of polynomials up to degree 99
# Our integrands are degree max 2+12+12=26
_NODES, _WEIGHTS = hermgauss(50)

# Precompute Hermite polynomial values at nodes for H0, H4, H8, H12
# H_n(x) is the physicist's Hermite polynomial
_H0 = np.ones_like(_NODES)
_H4 = hermval(_NODES, [0, 0, 0, 0, 16, 0, -48, 0, 12]) # 16x^4 - 48x^2 + 12
_H8 = hermval(_NODES, [0, 0, 0, 0, 0, 0, 0, 0, 256, 0, -4480, 0, 26880, 0, -53760, 0, 10752])
# H12 coefficients:
# H_12(x) = 4096 x^12 - 129024 x^10 + 1548288 x^8 - 7838304 x^6 + 15975360 x^4 - 12902400 x^2 + 3024000
_H12 = hermval(_NODES, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096, 0, -129024, 0, 1548288, 0, -7838304, 0, 15975360, 0, -12902400, 0, 3024000])

# Basis functions
_H_basis = np.vstack([_H0, _H4, _H8, _H12]) # Shape (4, N)

# Compute Moment Matrices M_k[i, j] = sum_n w_n x_n^k H_i(x_n) H_j(x_n)
# We need k=0 and k=2
_M0 = np.zeros((4, 4))
_M2 = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        _M0[i, j] = np.sum(_WEIGHTS * _H_basis[i] * _H_basis[j])
        _M2[i, j] = np.sum(_WEIGHTS * _NODES**2 * _H_basis[i] * _H_basis[j])

# For Fourier, Q(x) has coefficients a_i * (-1)^i
# N_k[i, j] = sum_n w_n x_n^k (-1)^i H_i(x_n) (-1)^j H_j(x_n) = (-1)^(i+j) M_k[i, j]
_signs = np.array([1.0, -1.0, 1.0, -1.0])
_N0 = _M0 * np.outer(_signs, _signs)
_N2 = _M2 * np.outer(_signs, _signs)

# Constraint g(0)=0: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# H0(0)=1, H4(0)=12, H8(0)=10752, H12(0)=3024000
# a12 = -(a0 + 12*a4 + 10752*a8) / 3024000
_da12 = np.array([-1.0/3024000.0, -12.0/3024000.0, -10752.0/3024000.0])


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _da12[0]*a0 + _da12[1]*a4 + _da12[2]*a8
    a = np.array([a0, a4, a8, a12])
    
    # Compute moments
    m0 = a @ _M0 @ a
    m2 = a @ _M2 @ a
    n0 = a @ _N0 @ a
    n2 = a @ _N2 @ a
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient
    # dm0/da = 2 * M0 @ a
    # dm2/da = 2 * M2 @ a
    # dn0/da = 2 * N0 @ a
    # dn2/da = 2 * N2 @ a
    
    dm0 = 2.0 * (_M0 @ a)
    dm2 = 2.0 * (_M2 @ a)
    dn0 = 2.0 * (_N0 @ a)
    dn2 = 2.0 * (_N2 @ a)
    
    # d(vf)/da = (dm2 * m0 - m2 * dm0) / m0^2
    dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
    
    # grad_bound = (vF * dvm2 + vf * dvn2) / (2 * bound)
    grad_full = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    # Project gradient to (a0, a4, a8)
    # a12 = da12 @ [a0, a4, a8]
    # da12/da_c = da12[c]
    # grad_c = grad_full[c] + grad_full[3] * da12[c]
    grad = np.zeros(3)
    for c in range(3):
        grad[c] = grad_full[c] + grad_full[3] * _da12[c]
        
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.0, 0.1, 0.001]),
        np.array([1.0, -0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([0.5, -0.05, 0.0005]),
        np.array([-2.0, 0.2, 0.002]),
        np.array([2.0, -0.2, 0.002]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal(0, 1.0, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.0, 0.1, 0.001])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
