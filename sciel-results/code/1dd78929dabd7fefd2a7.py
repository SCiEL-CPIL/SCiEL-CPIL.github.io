"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization.

g(x) = P(x) * exp(-x^2/2),  P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
a12 fixed so g(0) = 0.  Objective = sqrt( (m2/m0) * (n2/n0) ), the
variance product of |g|^2 and |G|^2 (G = Fourier transform, unitary
convention; the 2*pi factor in |G|^2 cancels in the ratio).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# ---- Precompute the moment tensors once (independent of the coefficients). ----
# I[n] = integral x^{2n} e^{-x^2} dx
def _fact_ratio(n):
    log_val = 0.0
    for m in range(1, 2 * n + 1):
        log_val += np.log(m)
    log_val -= n * np.log(4.0)
    for m in range(1, n + 1):
        log_val -= np.log(m)
    log_val += 0.5 * np.log(np.pi)
    return np.exp(log_val)

_I = np.array([_fact_ratio(n) for n in range(14)])
# T[k, i, j] = I[k + i + j]  (k in 0..2, i,j in 0..6)
_T = np.empty((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            _T[k, i, j] = _I[k + i + j]

# P(x) = sum_{k=0}^{6} p_k x^{2k}.  Columns are d p / d(a0,a4,a8) with the
# a12 = -(a0+3 a4+105 a8)/10395 constraint already substituted.
# H0=1; H4=x^4-6x^2+3; H8=x^8-28x^6+210x^4-420x^2+105;
# H12=x^12-66x^10+1155x^8-6930x^6+13860x^4-7560x^2+10395
# (constant term 10395 = 11!! enforces g(0)=0 when a12 = -(a0+3a4+105a8)/10395).
_C = np.array([
    [1.0,        3.0,       105.0],
    [0.0,       -6.0,      -420.0],
    [0.0,        1.0,       210.0],
    [0.0,        0.0,       -28.0],
    [0.0,        0.0,         1.0],
    [0.0,        0.0,         0.0],
    [0.0,        0.0,         0.0],
]) * 1.0
_C[0] += 945.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[1] += -7560.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[2] += 13860.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[3] += -6930.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[4] += 1155.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[5] += -66.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)
_C[6] += 1.0 * (np.array([1.0, 3.0, 105.0]) / 10395.0)

# S[k, i, c] = sum_j _C[i,c] * _T[k, i, j]   (used for the gradient)
_S = np.zeros((3, 7, 3))
for k in range(3):
    for i in range(7):
        for c in range(3):
            _S[k, i, c] = _C[i, c] * np.dot(_C[i, :], _T[k, i, :])

# q_k = p_k * (-1)^k  for the Fourier side
_SGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound_and_grad(params):
    p = np.array([params[0], params[1], params[2]], dtype=float)
    if not np.all(np.isfinite(p)):
        return 1e18, np.array([1e18, 1e18, 1e18])

    # p_k moments for |g|^2
    m0 = float(np.einsum('i,ij,j->', p, _T[0], p))
    m2 = float(np.einsum('i,ij,j->', p, _T[1], p))
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    # Fourier side: q_k = sgn_k * p_k
    q = _SGN * p
    n0 = float(np.einsum('i,ij,j->', q, _T[0], q))
    n2 = float(np.einsum('i,ij,j->', q, _T[1], q))
    if n0 <= 0 or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # gradient
    # d m_k / d a_c = 2 * sum_i _C[i,c] * (sum_j p_j T[k,i,j]) = 2 * sum_i _C[i,c] S[k,i,c]
    # d n_k / d a_c = 2 * sum_i _C[i,c]*sgn_i * (sum_j q_j T[k,i,j])
    grad = np.zeros(3)
    for c in range(3):
        dm0 = 2.0 * float(np.dot(_C[:, c], _S[0, :, c]))
        dm2 = 2.0 * float(np.dot(_C[:, c], _S[1, :, c]))
        # q-side derivative: d q_i / d a_c = sgn_i * _C[i,c]
        dq = _SGN * _C[:, c]
        dn0 = 2.0 * float(np.einsum('i,ij,j->', dq, _T[0], q))
        dn2 = 2.0 * float(np.einsum('i,ij,j->', dq, _T[1], q))

        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    return _bound_and_grad(params)[0]


def _objective_grad(params):
    return _bound_and_grad(params)


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    starts = [
        seed.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.3, 0.05, 0.0005]),
    ]
    rng = np.random.default_rng(123)
    for i in range(24):
        starts.append(rng.normal(0, 0.5, 3))

    best_val = np.inf
    best_params = None

    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 6000, 'ftol': 1e-16,
                                              'gtol': 1e-15, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Nelder-Mead polish from the best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 60000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 20000, 'ftol': 1e-17,
                                              'gtol': 1e-16, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return seed
    return np.array(best_params)
# EVOLVE-BLOCK-END
