"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute matrices for fast evaluation
# Basis indices: 0, 4, 8, 12
# Powers of x in H_n:
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients of H_n in powers of x^{2k}
# H_0: [1]
# H_4: [3, -6, 1] (x^0, x^2, x^4)
# H_8: [105, -420, 210, -28, 1] (x^0, x^2, x^4, x^6, x^8)
# H_12: [945, -7560, 13860, -6930, 1155, -66, 1] (x^0, ..., x^12)

# p vector has 7 components (powers 0 to 6 of x^2)
# p = [p_0, ..., p_6]
# p_k = sum_{basis} c_basis * coeff_{basis, k}

# c = [c_0, c_4, c_8, c_12]
# c_12 is determined by c_0, c_4, c_8 to satisfy g(0)=0.
# g(0) = c_0*1 + c_4*3 + c_8*105 + c_12*945 = 0
# c_12 = -(c_0 + 3c_4 + 105c_8) / 945

# Let a = [c_0, c_4, c_8]
# c = L a
# c_0 = a_0
# c_4 = a_1
# c_8 = a_2
# c_12 = -1/945 a_0 - 3/945 a_1 - 105/945 a_2

# Matrix L (4x3) mapping a to c
L_c = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/945.0, -3.0/945.0, -105.0/945.0]
])

# Matrix M (7x4) mapping c to p (coefficients of x^{2k})
# Rows k=0..6, Cols basis=0..3
M_p = np.zeros((7, 4))
# H_0
M_p[0, 0] = 1.0
# H_4
M_p[0, 1] = 3.0
M_p[1, 1] = -6.0
M_p[2, 1] = 1.0
# H_8
M_p[0, 2] = 105.0
M_p[1, 2] = -420.0
M_p[2, 2] = 210.0
M_p[3, 2] = -28.0
M_p[4, 2] = 1.0
# H_12
M_p[0, 3] = 945.0
M_p[1, 3] = -7560.0
M_p[2, 3] = 13860.0
M_p[3, 3] = -6930.0
M_p[4, 3] = 1155.0
M_p[5, 3] = -66.0
M_p[6, 3] = 1.0

# L (7x3) mapping a to p
L_p = M_p @ L_c

# Precompute I_n = integral x^{2n} e^{-x^2} dx
# I_n = (2n-1)!! / 2^n * sqrt(pi)
# I_0 = sqrt(pi)
# I_n = (2n-1)/2 * I_{n-1}
I = np.zeros(14)
I[0] = np.sqrt(np.pi)
for n in range(1, 14):
    I[n] = (2*n - 1) / 2.0 * I[n-1]

# Build B_k matrices (7x7)
# B_k[i, j] = I_{k + i + j}
# m_k = sum_{i,j} p_i p_j B_k[i,j] = p^T B_k p
B = [np.zeros((7,7)) for _ in range(3)]
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                B[k][i, j] = I[idx]

# G_k = L_p^T B_k L_p (3x3)
G = [L_p.T @ B[k] @ L_p for k in range(3)]

# Fourier side
# q_k = p_k (-1)^k
D = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
L_q = D @ L_p

# H_k = L_q^T B_k L_q (3x3)
# Note: n_k = 2*pi * q^T B_k q
# So H_k = 2*pi * L_q^T B_k L_q
H = [2.0 * np.pi * (L_q.T @ B[k] @ L_q) for k in range(3)]

def _bound_and_grad(params):
    a = params
    # m_k = a^T G_k a
    m0 = a @ G[0] @ a
    m2 = a @ G[1] @ a
    m4 = a @ G[2] @ a # not used in bound, but good for check
    
    n0 = a @ H[0] @ a
    n2 = a @ H[1] @ a
    n4 = a @ H[2] @ a

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # dm_k = 2 G_k a
    # dn_k = 2 H_k a
    dm0 = 2.0 * G[0] @ a
    dm2 = 2.0 * G[1] @ a
    dn0 = 2.0 * H[0] @ a
    dn2 = 2.0 * H[1] @ a
    
    # d(vf) = (dm2 * m0 - m2 * dm0) / m0^2
    # d(vF) = (dn2 * n0 - n2 * dn0) / n0^2
    dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
    
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
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
    
    # Initial points
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
    
    rng = np.random.default_rng(123)
    # Add random starts
    for i in range(50):
        # Scale: a0 ~ -1, a4 ~ 0.1, a8 ~ 0.001
        starts.append(rng.normal([0, 0, 0], [1.0, 0.1, 0.001], 3))
        
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
