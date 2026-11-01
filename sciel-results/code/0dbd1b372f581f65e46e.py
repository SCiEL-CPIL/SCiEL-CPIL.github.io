import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0

# Precomputation
# I[n] = integral x^{2n} e^{-x^2} dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/2
def _get_I(max_n):
    I = np.zeros(max_n + 1)
    I[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0
    return I

# We need I up to index 12 (for p_6 * p_6 * x^{12} * x^{12} -> x^{24} -> n=12? 
# Wait. m_k involves x^{2k} p_i p_j x^{2i} x^{2j} = x^{2(k+i+j)}.
# Max i,j is 6. Max k is 2. Max exponent is 2(2+6+6) = 28.
# So we need I up to 14.
I_vals = _get_I(14)

# Matrices A_k (7x7) where A_k[i,j] = I[k+i+j]
# p is 7x1 vector.
A = []
B = []
D = np.diag([1, -1, 1, -1, 1, -1, 1])
for k in range(3):
    A_k = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            A_k[i, j] = I_vals[k + i + j]
    A.append(A_k)
    # B_k = 2*pi * D * A_k * D
    B.append(2.0 * np.pi * (D @ A_k @ D))

# Jacobian of p w.r.t a = [a0, a4, a8]
# p0 = 0 (always)
# p1 = -6 a4 - 420 a8 - 7560 a12
# p2 = a4 + 210 a8 + 13860 a12
# p3 = -28 a8 - 6930 a12
# p4 = a8 + 1155 a12
# p5 = -66 a12
# p6 = a12
# a12 = -(a0 + 3 a4 + 105 a8) / 945

# Let's compute J explicitly.
# da12/da = [-1/945, -3/945, -105/945]
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# p1 coeffs: [0, -6, -420] + [-7560] * c12
J = np.zeros((7, 3))
J[0] = 0.0
J[1] = np.array([0.0, -6.0, -420.0]) - 7560.0 * c12
J[2] = np.array([0.0, 1.0, 210.0]) + 13860.0 * c12
J[3] = np.array([0.0, 0.0, -28.0]) - 6930.0 * c12
J[4] = np.array([0.0, 0.0, 1.0]) + 1155.0 * c12
J[5] = np.array([0.0, 0.0, 0.0]) - 66.0 * c12
J[6] = np.array([0.0, 0.0, 0.0]) + 1.0 * c12

def _bound_and_grad(a):
    a0, a4, a8 = a
    # Compute p
    # p = J @ a
    p = J @ a
    
    # Compute m_k = p^T A_k p
    m = np.array([p @ A[k] @ p for k in range(3)])
    # Compute n_k = p^T B_k p
    n = np.array([p @ B[k] @ p for k in range(3)])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-18 or n0 <= 1e-18 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vf * vF)
    
    # Gradient w.r.t p
    # d(bound)/dp = (1/(2*bound)) * (vF * d(vf)/dp + vf * d(vF)/dp)
    # d(vf)/dp = (dm2/dp * m0 - m2 * dm0/dp) / m0^2
    # dm_k/dp = 2 A_k p
    
    dm_dp = 2.0 * np.array([A[k] @ p for k in range(3)])
    dn_dp = 2.0 * np.array([B[k] @ p for k in range(3)])
    
    dvm2_dp = (dm_dp[1] * m0 - m2 * dm_dp[0]) / (m0 * m0)
    dvn2_dp = (dn_dp[1] * n0 - n2 * dn_dp[0]) / (n0 * n0)
    
    grad_p = (vF * dvm2_dp + vf * dvn2_dp) / (2.0 * bound)
    
    # Gradient w.r.t a
    # grad_a = J^T grad_p
    grad_a = J.T @ grad_p
    
    return bound, grad_a

def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    best_val, _ = _bound_and_grad(seed)
    best_params = seed.copy()
    
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [seed]
    
    # Local perturbations
    for _ in range(20):
        starts.append(seed + rng.normal(0, 0.1, 3))
        
    # Global random points
    for _ in range(40):
        starts.append(rng.normal(0, 1.0, 3))
        
    # Specific structured points
    starts.append(np.array([-1.0, 0.0, 0.0]))
    starts.append(np.array([0.0, -0.1, 0.0]))
    starts.append(np.array([0.0, 0.0, -0.001]))
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_bound_and_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_bound_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
    # Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(lambda x: _bound_and_grad(x)[0], best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
