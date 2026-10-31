# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, sqrt, pi

_DEADLINE = 180.0

def _hermite_coeffs(n):
    if n == 0: return {0: 1.0}
    if n == 1: return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            new_p = p + 1
            h_new[new_p] = h_new.get(new_p, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr

def _int_x2m(m):
    if m == 0: return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

# Degrees of freedom for p(x)
DEGREES = [0, 4, 8, 12]
N_DEG = 4

# Precompute Hermite polynomials as coefficient vectors
H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

# Precompute M0, M2, L matrices
M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)

M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        deg_i = DEGREES[i]
        deg_j = DEGREES[j]
        ci = H_POLYS[i]
        cj = H_POLYS[j]
        prod = np.convolve(ci, cj)
        prod_x2 = np.zeros(len(prod) + 2)
        prod_x2[2:] = prod
        val = 0.0
        for k in range(0, len(prod_x2), 2):
            m = k // 2
            val += prod_x2[k] * _int_x2m(m)
        M2_mat[i, j] = val

Q_POLYS = []
for i, d in enumerate(DEGREES):
    q_vec = np.zeros(max(d + 1, 1))
    if d == 0:
        q_vec = np.zeros(2)
        q_vec[1] = -1.0
    else:
        h_minus = _hermite_coeffs(d - 1)
        h_plus = _hermite_coeffs(d + 1)
        max_deg_q = d + 1
        q_vec = np.zeros(max_deg_q + 1)
        for p, c in h_minus.items():
            q_vec[p] += d * c
        for p, c in h_plus.items():
            q_vec[p] -= 0.5 * c
    Q_POLYS.append(q_vec)

L_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci = Q_POLYS[i]
        cj = Q_POLYS[j]
        prod = np.convolve(ci, cj)
        val = 0.0
        for k in range(0, len(prod), 2):
            m = k // 2
            val += prod[k] * _int_x2m(m)
        L_mat[i, j] = val

# Construct the transformation matrix B such that a_4 = B @ a_3
# a_3 = [a0, a4, a8]
# a_12 = -(a0 + 12*a4 + 1680*a8) / 665280
# a_4_vec = [a0, a4, a8, a12]
# B = [[1, 0, 0],
#      [0, 1, 0],
#      [0, 0, 1],
#      [-1/665280, -12/665280, -1680/665280]]

B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/665280.0, -12.0/665280.0, -1680.0/665280.0]
])

# Effective 3x3 matrices
M0_eff = B.T @ M0_mat @ B
M2_eff = B.T @ M2_mat @ B
L_eff = B.T @ L_mat @ B

def _compute_bound_and_grad_a3(a3):
    """
    Computes bound and gradient w.r.t a3.
    Returns (bound, grad_a3, valid)
    """
    M0 = a3 @ M0_eff @ a3
    M2 = a3 @ M2_eff @ a3
    L = a3 @ L_eff @ a3
    
    if M0 <= 1e-12 or L <= 1e-12 or M2 <= 1e-12:
        return 1e10, np.zeros(3), False
    
    bound = np.sqrt(M2 * L) / M0
    
    # Gradient w.r.t a3
    dM0 = 2 * M0_eff @ a3
    dM2 = 2 * M2_eff @ a3
    dL = 2 * L_eff @ a3
    
    # d(bound)/da3 = d/dM2 * dM2 + d/dL * dL - d/dM0 * dM0
    # bound = (M2 L)^0.5 / M0
    # db/dM2 = 0.5 * (M2 L)^-0.5 * L / M0
    # db/dL = 0.5 * (M2 L)^-0.5 * M2 / M0
    # db/dM0 = - (M2 L)^0.5 / M0^2
    
    term_M2 = (0.5 * L) / (M0 * np.sqrt(M2 * L))
    term_L = (0.5 * M2) / (M0 * np.sqrt(M2 * L))
    term_M0 = -bound / M0
    
    grad_a3 = term_M2 * dM2 + term_L * dL + term_M0 * dM0
    
    return bound, grad_a3, True

def _objective_angles(params):
    theta, phi = params
    # a = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
    a3 = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    bound, _, valid = _compute_bound_and_grad_a3(a3)
    if not valid:
        return 1e10
    return bound

def _gradient_angles(params):
    theta, phi = params
    a3 = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    bound, grad_a3, valid = _compute_bound_and_grad_a3(a3)
    if not valid:
        return np.zeros(2)
    
    # Jacobian of a3 w.r.t [theta, phi]
    # a0 = sin(t)cos(p) -> da0/dt = cos(t)cos(p), da0/dp = -sin(t)sin(p)
    # a4 = sin(t)sin(p) -> da4/dt = cos(t)sin(p), da4/dp = sin(t)cos(p)
    # a8 = cos(t)       -> da8/dt = -sin(t),      da8/dp = 0
    
    jac = np.array([
        [np.cos(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi)],
        [np.cos(theta) * np.sin(phi),  np.sin(theta) * np.cos(phi)],
        [-np.sin(theta),               0.0]
    ])
    
    # grad_params = jac.T @ grad_a3
    grad_params = jac.T @ grad_a3
    return grad_params

def construct():
    start_time = time.time()
    
    # Grid search for initial guess
    thetas = np.linspace(0.01, np.pi - 0.01, 50)
    phis = np.linspace(0.0, 2.0 * np.pi, 100)
    
    best_val = np.inf
    best_params = None
    
    for t in thetas:
        for p in phis:
            if time.time() - start_time > _DEADLINE - 50:
                break
            val = _objective_angles([t, p])
            if val < best_val:
                best_val = val
                best_params = [t, p]
        if time.time() - start_time > _DEADLINE - 50:
            break
            
    if best_params is None:
        # Fallback to previous best guess
        a3_init = np.array([-1.13, 0.04, 0.0003])
        # Convert to angles? Or just return a3_init directly?
        # The interface expects coefficients.
        return a3_init
        
    # Refine with minimize
    bounds = [(0.0, np.pi), (0.0, 2.0 * np.pi)]
    
    try:
        result = minimize(
            _objective_angles,
            best_params,
            jac=_gradient_angles,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
        
    # Try a few random starts to avoid local minima
    np.random.seed(42)
    for _ in range(10):
        if time.time() - start_time > _DEADLINE - 20:
            break
        t_rand = np.random.uniform(0.01, np.pi - 0.01)
        p_rand = np.random.uniform(0.0, 2.0 * np.pi)
        try:
            res = minimize(
                _objective_angles,
                [t_rand, p_rand],
                jac=_gradient_angles,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Convert best angles back to coefficients
    theta, phi = best_params
    a3 = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    return a3

# EVOLVE-BLOCK-END
