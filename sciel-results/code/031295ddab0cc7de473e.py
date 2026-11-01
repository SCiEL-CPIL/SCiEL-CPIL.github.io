"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
DEADLINE = 210.0
SAFETY_MARGIN = 15.0

def _check_deadline():
    return (time.time() - _START_TIME) < (DEADLINE - SAFETY_MARGIN)

# Precompute I_n = integral x^{2n} e^{-x^2} dx
# I_0 = sqrt(pi), I_n = (2n-1)/2 * I_{n-1}
_I_max = 25
_I = np.zeros(_I_max + 1)
_I[0] = np.sqrt(np.pi)
for n in range(1, _I_max + 1):
    _I[n] = (2.0 * n - 1.0) / 2.0 * _I[n - 1]

# Coefficients for the polynomial P(x) = sum_{k=0}^6 p_k x^{2k}
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
# a12 = -(a0 + 3*a4 + 105*a8) / 945.0

# Matrix A such that p = A @ [a0, a4, a8, a12]
# p0: 1, 3, 105, 945
# p1: 0, -6, -420, -7560
# p2: 0, 1, 210, 13860
# p3: 0, 0, -28, -6930
# p4: 0, 0, 1, 1155
# p5: 0, 0, 0, -66
# p6: 0, 0, 0, 1
_A = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

# We can reduce the dimensionality by substituting a12.
# p = A[:, :3] @ [a0, a4, a8] + A[:, 3:4] @ [a12]
# a12 = -1/945 * a0 - 3/945 * a4 - 105/945 * a8
# Let M = A[:, :3] - A[:, 3:4] @ [1/945, 3/945, 105/945]
# Then p = M @ [a0, a4, a8]
_M = _A[:, :3] - _A[:, 3:4] * np.array([1.0/945.0, 3.0/945.0, 105.0/945.0])

# Precompute the moment matrices
# m_k = sum_{i,j} p_i p_j I[k+i+j]
# n_k = 2pi * sum_{i,j} q_i q_j I[k+i+j] where q_i = p_i (-1)^i

# Let D = diag((-1)^i) for i=0..6
_D = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute G_k such that m_k = p^T G_k p
# G_k[i,j] = I[k+i+j]
_G = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx <= _I_max:
                _G[k, i, j] = _I[idx]

# Precompute H_k such that n_k = 2pi * q^T H_k q = 2pi * p^T D H_k D p
# H_k[i,j] = I[k+i+j] is the same structure, but q = D p.
# n_k = 2pi * (D p)^T G_k (D p) = 2pi * p^T D G_k D p
_H = np.zeros((3, 7, 7))
for k in range(3):
    _H[k] = _D @ _G[k] @ _D

# Now p = M @ a, where a = [a0, a4, a8]
# m_k = a^T M^T G_k M a
# n_k = 2pi * a^T M^T H_k M a

_MT_G = np.zeros((3, 3, 3))
_MT_H = np.zeros((3, 3, 3))
for k in range(3):
    _MT_G[k] = _M.T @ _G[k] @ _M
    _MT_H[k] = _M.T @ _H[k] @ _M

def _compute_objective_and_grad(a):
    """
    Compute the uncertainty product bound and its gradient.
    a: array of shape (3,)
    Returns: (value, grad)
    """
    a0, a4, a8 = a
    # Moments
    m0 = np.dot(a, np.dot(_MT_G[0], a))
    m2 = np.dot(a, np.dot(_MT_G[1], a))
    
    n0 = 2.0 * np.pi * np.dot(a, np.dot(_MT_H[0], a))
    n2 = 2.0 * np.pi * np.dot(a, np.dot(_MT_H[1], a))
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # dm0 = 2 * G0 @ a
    # dm2 = 2 * G1 @ a
    # dn0 = 2pi * 2 * H0 @ a
    # dn2 = 2pi * 2 * H1 @ a
    
    dm0 = 2.0 * _MT_G[0] @ a
    dm2 = 2.0 * _MT_G[1] @ a
    dn0 = 4.0 * np.pi * _MT_H[0] @ a
    dn2 = 4.0 * np.pi * _MT_H[1] @ a
    
    # d(vf) = (dm2 * m0 - m2 * dm0) / m0^2
    # d(vF) = (dn2 * n0 - n2 * dn0) / n0^2
    
    dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
    
    # d(bound) = (vF * dvm2 + vf * dvn2) / (2 * bound)
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective_val(a):
    val, _ = _compute_objective_and_grad(a)
    return val

def construct():
    if not _check_deadline():
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf
    
    # Seed point
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # 1. Local optimization from seed
    try:
        result = minimize(_compute_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
        if result.fun < best_score and np.isfinite(result.fun):
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
    
    # 2. Global search with Differential Evolution
    # Wider bounds to explore the landscape
    bounds = [(-10, 10), (-5, 5), (-1, 1)]
    try:
        de_result = differential_evolution(_objective_val, bounds, 
                                           maxiter=300, popsize=30,
                                           seed=42, tol=1e-12,
                                           polish=True, 
                                           init='sobol',
                                           updating='deferred')
        if de_result.fun < best_score and np.isfinite(de_result.fun):
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
        
    # 3. Multi-start local refinement
    # Generate starts around the best found and the seed
    rng = np.random.default_rng(42)
    starts = [x0]
    if best_result is not None:
        starts.append(best_result)
        
    # Add random perturbations
    for _ in range(50):
        if best_result is not None:
            base = best_result
        else:
            base = x0
        pert = base + rng.normal(0, 0.5, 3)
        starts.append(pert)
        
    # Structured starts
    for scale in [0.1, 0.5, 2.0, 5.0]:
        starts.append(x0 * scale)
        if best_result is not None:
            starts.append(best_result * scale)

    for start in starts:
        if not _check_deadline():
            break
        try:
            res = minimize(_compute_objective_and_grad, start, method='L-BFGS-B', jac=True,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
    return x0

# EVOLVE-BLOCK-END
