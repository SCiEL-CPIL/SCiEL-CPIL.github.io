"""Analytic uncertainty bound with vectorized gradient + multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx
def _precompute_I():
    I = np.zeros(20)
    for n in range(20):
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute moment matrices M_k and N_k
# M_k[i,j] = I_{k+i+j}
# N_k[i,j] = I_{k+i+j} (same, but q = D p)
M0 = np.zeros((7, 7))
M1 = np.zeros((7, 7))
M2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        M0[i,j] = _I[i+j]
        M1[i,j] = _I[i+j+1]
        M2[i,j] = _I[i+j+2]

# Signs for q
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_D = np.diag(_signs)
_pi2 = 2.0 * np.pi

# Mapping from (a0, a4, a8) to p
# p0 = a0 + 3a4 + 105a8 + 945a12, a12 = -(a0 + 3a4 + 105a8)/945
# p0 = 0
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12

# Let's construct B such that p = B @ a
# a = [a0, a4, a8]
# a12 = -1/945 a0 - 3/945 a4 - 105/945 a8

B = np.zeros((7, 3))
# Row 0: p0 = 0 (always, due to constraint)
# Row 1: p1 = -6a4 - 420a8 - 7560(-1/945 a0 - 3/945 a4 - 105/945 a8)
#        = 8 a0 + (-6 + 24) a4 + (-420 + 840) a8
#        = 8 a0 + 18 a4 + 420 a8
B[1, 0] = 8.0
B[1, 1] = 18.0
B[1, 2] = 420.0

# Row 2: p2 = a4 + 210a8 + 13860(-1/945 a0 - 3/945 a4 - 105/945 a8)
#        = -14.666 a0 + (1 - 44) a4 + (210 - 1540) a8
#        = -14.666 a0 - 43 a4 - 1330 a8
B[2, 0] = -13860.0 / 945.0
B[2, 1] = 1.0 - 3*13860.0 / 945.0
B[2, 2] = 210.0 - 105*13860.0 / 945.0

# Row 3: p3 = -28a8 - 6930(-1/945 a0 - 3/945 a4 - 105/945 a8)
#        = 7.333 a0 + 22 a4 + (-28 + 770) a8
#        = 7.333 a0 + 22 a4 + 742 a8
B[3, 0] = 6930.0 / 945.0
B[3, 1] = 3*6930.0 / 945.0
B[3, 2] = -28.0 + 105*6930.0 / 945.0

# Row 4: p4 = a8 + 1155(-1/945 a0 - 3/945 a4 - 105/945 a8)
#        = -1.222 a0 - 3.666 a4 + (1 - 128.33) a8
B[4, 0] = -1155.0 / 945.0
B[4, 1] = -3*1155.0 / 945.0
B[4, 2] = 1.0 - 105*1155.0 / 945.0

# Row 5: p5 = -66(-1/945 a0 - 3/945 a4 - 105/945 a8)
B[5, 0] = 66.0 / 945.0
B[5, 1] = 3*66.0 / 945.0
B[5, 2] = 105*66.0 / 945.0

# Row 6: p6 = -1/945 a0 - 3/945 a4 - 105/945 a8
B[6, 0] = -1.0 / 945.0
B[6, 1] = -3.0 / 945.0
B[6, 2] = -105.0 / 945.0

# Verify p0 = 0
# p0 = a0 + 3a4 + 105a8 + 945a12 = a0 + 3a4 + 105a8 - (a0 + 3a4 + 105a8) = 0. Correct.

def _bound_and_grad(params):
    a = np.array(params)
    p = B @ a
    q = _signs * p
    
    m0 = p @ M0 @ p
    m1 = p @ M1 @ p
    n0 = _pi2 * (q @ M0 @ q)
    n1 = _pi2 * (q @ M1 @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # dm0/da = B^T (2 M0 p)
    # dm1/da = B^T (2 M1 p)
    # dn0/da = B^T (2 _pi2 D M0 q)
    # dn1/da = B^T (2 _pi2 D M1 q)
    
    dm0_da = 2.0 * B.T @ (M0 @ p)
    dm1_da = 2.0 * B.T @ (M1 @ p)
    dn0_da = 2.0 * _pi2 * B.T @ (_D @ (M0 @ q))
    dn1_da = 2.0 * _pi2 * B.T @ (_D @ (M1 @ q))
    
    dvm2_da = (m0 * dm1_da - m1 * dm0_da) / (m0 * m0)
    dvn2_da = (n0 * dn1_da - n1 * dn0_da) / (n0 * n0)
    
    grad = (vF * dvm2_da + vf * dvn2_da) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    best_val = np.inf
    best_params = None
    
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        scale = 0.2 + 0.5 * i / 20
        starts.append(rng.normal(0, scale, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
