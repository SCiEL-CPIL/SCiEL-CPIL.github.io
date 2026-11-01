"""
Optimized uncertainty bound minimization using vectorized computations and global search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Set a safe deadline (200s out of 210s limit)
_DEADLINE = time.time() + 200

def _time_left():
    return _DEADLINE - time.time()

# Precompute constants
# I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute the derivative coefficient matrix
# p_k coefficients in terms of a0, a4, a8 (a12 is dependent)
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12
# a12 = -(a0 + 3a4 + 105a8)/945
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Matrix M such that p = M @ [a0, a4, a8]
# Row k corresponds to p_k
_M = np.zeros((7, 3))
_M[0] = [1.0 + 945.0*_da12[0], 3.0 + 945.0*_da12[1], 105.0 + 945.0*_da12[2]]
_M[1] = [0.0 - 7560.0*_da12[0], -6.0 - 7560.0*_da12[1], -420.0 - 7560.0*_da12[2]]
_M[2] = [0.0 + 13860.0*_da12[0], 1.0 + 13860.0*_da12[1], 210.0 + 13860.0*_da12[2]]
_M[3] = [0.0 - 6930.0*_da12[0], 0.0 - 6930.0*_da12[1], -28.0 - 6930.0*_da12[2]]
_M[4] = [0.0 + 1155.0*_da12[0], 0.0 + 1155.0*_da12[1], 1.0 + 1155.0*_da12[2]]
_M[5] = [0.0 - 66.0*_da12[0], 0.0 - 66.0*_da12[1], 0.0 - 66.0*_da12[2]]
_M[6] = [0.0 + 1.0*_da12[0], 0.0 + 1.0*_da12[1], 0.0 + 1.0*_da12[2]]

# Signs for q_k = (-1)^k p_k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute I matrices for k=0, 1, 2
# I_k[i,j] = I[i+j+k]
_I0 = np.zeros((7, 7))
_I1 = np.zeros((7, 7))
_I2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _I0[i, j] = _I[i+j]
        _I1[i, j] = _I[i+j+1]
        _I2[i, j] = _I[i+j+2]

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    Vectorized implementation.
    """
    a = np.array(params)
    p = _M @ a
    q = p * _signs
    
    # m_k = p^T I_k p
    m0 = np.dot(p, _I0 @ p)
    m1 = np.dot(p, _I1 @ p)
    m2 = np.dot(p, _I2 @ p)
    
    # n_k = 2*pi * q^T I_k q
    two_pi = 2.0 * np.pi
    n0 = two_pi * np.dot(q, _I0 @ q)
    n1 = two_pi * np.dot(q, _I1 @ q)
    n2 = two_pi * np.dot(q, _I2 @ q)
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
        
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 1e-15 or vF <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm1*m0 - m1*dm0) / m0^2
    # dm_k/da_c = 2 * (M_c)^T I_k p  where M_c is the c-th column of M
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _M[:, c]
        dq = dp * _signs
        
        # dm_k/da_c = 2 * dp^T I_k p
        dm0 = 2.0 * np.dot(dp, _I0 @ p)
        dm1 = 2.0 * np.dot(dp, _I1 @ p)
        dm2 = 2.0 * np.dot(dp, _I2 @ p)
        
        # dn_k/da_c = 2 * two_pi * dq^T I_k q
        dn0 = 2.0 * two_pi * np.dot(dq, _I0 @ q)
        dn1 = 2.0 * two_pi * np.dot(dq, _I1 @ q)
        dn2 = 2.0 * two_pi * np.dot(dq, _I2 @ q)
        
        dvm2 = (dm1*m0 - m1*dm0) / (m0*m0)
        dvn2 = (dn1*n0 - n1*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642]) # Fallback seed
    
    # 1. Global Search with Differential Evolution
    # Bounds: a0 in [-5, 5], a4 in [-1, 1], a8 in [-0.1, 0.1]
    # The seed is around (-1.13, 0.04, 0.0003)
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    if _time_left() > 120:
        try:
            de_result = differential_evolution(
                _objective, bounds, seed=42, maxiter=50, popsize=20,
                tol=1e-10, init='sobol', polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x
        except Exception:
            pass

    # 2. Local Polish with L-BFGS-B
    # Use multiple starting points: the DE result, the seed, and some random perturbations
    starts = [best_params, np.array([-1.13, 0.04, 0.000308642])]
    
    # Add some random starts around the seed and DE result
    rng = np.random.default_rng(42)
    for _ in range(5):
        if _time_left() < 20: break
        # Perturb around current best
        perturb = rng.normal(0, 0.1, 3)
        starts.append(best_params + perturb)
        
    for start in starts:
        if _time_left() < 10: break
        try:
            res = minimize(
                _objective_grad, start, method='L-BFGS-B', jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # 3. Final High-Precision Polish
    if _time_left() > 5:
        try:
            res = minimize(
                _objective_grad, best_params, method='L-BFGS-B', jac=True,
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END
