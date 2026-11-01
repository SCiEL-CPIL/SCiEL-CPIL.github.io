"""
Optimized search for Fourier-invariant Hermite-Gaussian test function coefficients.
Uses differential evolution for global exploration and L-BFGS-B with analytic gradients for local refinement.
Parameters are scaled to O(1) for better optimizer performance.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_DEADLINE = time.time() + 195  # Safety margin before 210s kill

def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    Physicists' Hermite polynomials.
    H_n(x) represented as vector of coefficients for x^0, x^2, ..., x^{2n}.
    Here we align them to a common basis of x^0, x^2, ..., x^12 (7 terms).
    """
    # Basis: x^0, x^2, x^4, x^6, x^8, x^10, x^12
    H = np.zeros((4, 7))
    # H_0 = 1
    H[0, 0] = 1.0
    # H_4 = x^4 - 6x^2 + 3
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    # H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    # H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H

def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, ..., 12.
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I

_H = _hermite_coeffs()
_I = _compute_moments()

# Precompute the matrices for the quadratic forms
# We express P(x) = sum_{k=0}^6 p_k x^{2k}.
# |P(x)|^2 = sum_{i,j} p_i p_j x^{2(i+j)}.
# m_k = int x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j].
# We need m_0, m_2 for A(f)^2 = m_2/m_0.
# For the Fourier side, Q(xi) = P(-i xi). Coeffs q_k = p_k (-1)^k.
# |Q|^2 = sum q_i q_j xi^{2(i+j)}.
# n_k = int xi^{2k} |Q|^2 e^{-xi^2} dx = sum q_i q_j I[k+i+j].
# A(F)^2 = n_2/n_0. Note: The Fourier transform definition might introduce factors of 2pi or 1/sqrt(2pi).
# The seed code uses G(xi) = sqrt(2pi) P(-i xi) e^{-xi^2/2}.
# So |G|^2 = 2pi |P(-i xi)|^2 e^{-xi^2}.
# Thus n_k in the seed code includes the 2pi factor.
# Let's build the matrices M0, M2, N0, N2 such that:
# m0 = p^T M0 p, m2 = p^T M2 p
# n0 = p^T N0 p, n2 = p^T N2 p (where N includes the 2pi factor)

# p is 7x1.
# M0[i,j] = I[i+j]
# M2[i,j] = I[1+i+j]
# q_k = p_k (-1)^k. Let D = diag(1, -1, 1, -1, 1, -1, 1).
# q = D p.
# n0 = 2pi q^T N0_raw q = 2pi p^T D N0_raw D p.
# Let N0_mat = 2pi * D @ N0_raw @ D.
# N0_raw[i,j] = I[i+j].
# N2_raw[i,j] = I[1+i+j].
# N2_mat = 2pi * D @ N2_raw @ D.

M0 = np.zeros((7, 7))
M2 = np.zeros((7, 7))
N0_raw = np.zeros((7, 7))
N2_raw = np.zeros((7, 7))

for i in range(7):
    for j in range(7):
        M0[i, j] = _I[i + j]
        M2[i, j] = _I[1 + i + j]
        N0_raw[i, j] = _I[i + j]
        N2_raw[i, j] = _I[1 + i + j]

D = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
N0 = 2.0 * np.pi * (D @ N0_raw @ D)
N2 = 2.0 * np.pi * (D @ N2_raw @ D)

# The constraint P(0) = 0 implies sum_{k} a_m H[m,0] = 0?
# P(0) = a0 H0(0) + a4 H4(0) + a8 H8(0) + a12 H12(0) = 0.
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945.
# a0 + 3a4 + 105a8 + 945a12 = 0 => a12 = -(a0 + 3a4 + 105a8)/945.

# We optimize over a0, a4, a8.
# p = a0 H[0] + a4 H[1] + a8 H[2] + a12 H[3]
# p = a0 (H[0] - (1/945)H[3]) + a4 (H[1] - (3/945)H[3]) + a8 (H[2] - (105/945)H[3])
# Let v0 = H[0] - (1/945)H[3]
#     v4 = H[1] - (3/945)H[3]
#     v8 = H[2] - (105/945)H[3]
# p = a0 v0 + a4 v4 + a8 v8

v0 = _H[0] - (1.0 / 945.0) * _H[3]
v4 = _H[1] - (3.0 / 945.0) * _H[3]
v8 = _H[2] - (105.0 / 945.0) * _H[3]
V = np.column_stack((v0, v4, v8))  # 7x3 matrix

# p = V @ a, where a = [a0, a4, a8]^T
# m0 = a^T (V^T M0 V) a
# m2 = a^T (V^T M2 V) a
# n0 = a^T (V^T N0 V) a
# n2 = a^T (V^T N2 V) a

Am0 = V.T @ M0 @ V
Am2 = V.T @ M2 @ V
An0 = V.T @ N0 @ V
An2 = V.T @ N2 @ V

# Objective: minimize sqrt( (a^T Am2 a / a^T Am0 a) * (a^T An2 a / a^T An0 a) )
# Equivalent to minimizing (a^T Am2 a * a^T An2 a) / (a^T Am0 a * a^T An0 a)
# Let f(a) = (a^T Am2 a * a^T An2 a) / (a^T Am0 a * a^T An0 a)
# We minimize sqrt(f(a)).

def _objective_and_grad(a_scaled):
    """
    a_scaled: [u0, u4, u8]
    a0 = u0
    a4 = u4 * 0.04
    a8 = u8 * 0.0003
    """
    a0 = a_scaled[0]
    a4 = a_scaled[1] * 0.04
    a8 = a_scaled[2] * 0.0003
    a = np.array([a0, a4, a8])
    
    m0 = a @ Am0 @ a
    m2 = a @ Am2 @ a
    n0 = a @ An0 @ a
    n2 = a @ An2 @ a
    
    if m0 < 1e-15 or n0 < 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    
    if m2 < 0 or n2 < 0:
        return 1e18, np.zeros(3)
        
    # f = (m2 * n2) / (m0 * n0)
    # val = sqrt(f)
    # grad_val = 0.5 * f^-0.5 * grad_f
    # grad_f = (grad(m2*n2) * m0*n0 - m2*n2 * grad(m0*n0)) / (m0*n0)^2
    
    val2 = (m2 * n2) / (m0 * n0)
    if val2 < 0:
        return 1e18, np.zeros(3)
    val = np.sqrt(val2)
    
    # Gradient of m_k = a^T M_k a is 2 M_k a (symmetric)
    dm0 = 2.0 * (Am0 @ a)
    dm2 = 2.0 * (Am2 @ a)
    dn0 = 2.0 * (An0 @ a)
    dn2 = 2.0 * (An2 @ a)
    
    # d(m2*n2)/da = dm2 * n2 + m2 * dn2
    # d(m0*n0)/da = dm0 * n0 + m0 * dn0
    
    num = m2 * n2
    den = m0 * n0
    
    d_num = dm2 * n2 + m2 * dn2
    d_den = dm0 * n0 + m0 * dn0
    
    # d(val2) = (d_num * den - num * d_den) / den^2
    d_val2 = (d_num * den - num * d_den) / (den * den)
    
    # d(val) = 0.5 * val2^-0.5 * d_val2
    d_val = 0.5 / val * d_val2
    
    # d_val is gradient with respect to a = [a0, a4, a8]
    # We need gradient with respect to a_scaled = [u0, u4, u8]
    # a0 = u0 -> da0/du0 = 1
    # a4 = 0.04 u4 -> da4/du4 = 0.04
    # a8 = 0.0003 u8 -> da8/du8 = 0.0003
    grad_scaled = np.array([
        d_val[0] * 1.0,
        d_val[1] * 0.04,
        d_val[2] * 0.0003
    ])
    
    return val, grad_scaled

def _objective(a_scaled):
    val, _ = _objective_and_grad(a_scaled)
    return val

def _objective_grad(a_scaled):
    return _objective_and_grad(a_scaled)

def construct():
    """
    Multi-stage optimization:
    1. Differential Evolution for global search.
    2. L-BFGS-B for local refinement.
    3. Final high-precision polish.
    """
    from scipy.optimize import minimize, differential_evolution
    
    # Bounds for u0, u4, u8
    # a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003
    # u0 = a0, u4 = a4/0.04, u8 = a8/0.0003
    # Typical ranges: a0 in [-5, 5], a4 in [-0.2, 0.2], a8 in [-0.002, 0.002]
    # u0 in [-5, 5]
    # u4 in [-5, 5]
    # u8 in [-6.6, 6.6]
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-6.6, 6.6)]
    
    best_val = np.inf
    best_params_scaled = np.array([0.0, 0.0, 0.0])
    
    # Stage 1: Differential Evolution
    # This is slower but more robust for global optima.
    if time.time() < _DEADLINE - 30:
        try:
            de_result = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.9
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params_scaled = de_result.x
        except Exception as e:
            pass
    
    # Stage 2: Local refinement from DE result and some heuristic starts
    starts = [best_params_scaled.copy()]
    
    # Heuristic starts based on seed values
    # Seed: a0=-1.13, a4=0.04, a8=0.000308642
    # u0=-1.13, u4=1.0, u8=1.0288
    starts.append(np.array([-1.13, 1.0, 1.0288]))
    starts.append(np.array([-1.0, 0.5, 0.5]))
    starts.append(np.array([-1.5, 1.5, 1.5]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(
                _objective_grad, 
                start, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_scaled = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(
                _objective_grad, 
                best_params_scaled, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params_scaled = res.x.copy()
        except Exception:
            pass
            
    # Convert back to a0, a4, a8
    a0 = best_params_scaled[0]
    a4 = best_params_scaled[1] * 0.04
    a8 = best_params_scaled[2] * 0.0003
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
