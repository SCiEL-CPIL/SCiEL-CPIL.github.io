# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 615

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    X = np.zeros((N_pts_ct * N_pts_phi, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _precompute_tensors(X, w_flat):
    """Precompute Q (2x2) and T (4x4x4x4) tensors for spectral L2/L4 norm computation."""
    Q = X.T @ (w_flat[:, None] * X)
    T = np.einsum('pi,pj,pk,pl,p->ijkl', X, X, X, X, w_flat, optimize=True)
    return Q, T

def _ratio_and_grad_spectral(c, Q, T):
    """Compute R = ||f||_4 / ||f||_2 and gradient using precomputed tensors."""
    S2 = c @ Q @ c
    if S2 < 1e-10:
        return 1e10, np.zeros(21)
    
    S4 = np.einsum('ijkl,i,j,k,l->', T, c, c, c, c, optimize=True)
    if S4 < 1e-10:
        return 1e10, np.zeros(21)
    
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    grad_S4_over_4 = np.einsum('ijkl,j,k,l->i', T, c, c, c, optimize=True)
    grad_S2_over_2 = Q @ c
    
    grad_log_R = grad_S4_over_4 / S4 - grad_S2_over_2 / S2
    grad_R = R * grad_log_R
    
    return R, grad_R

def _sphere_projected_minimize(start, Q, T, maxiter=500):
    """Minimize R on the unit sphere using L-BFGS-B with sphere projection."""
    def obj_and_grad(u):
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            return 1e10, np.zeros(21)
        c = u / norm_u
        R, grad_R = _ratio_and_grad_spectral(c, Q, T)
        proj = grad_R - np.dot(grad_R, c) * c
        grad_u = proj / norm_u
        return R, grad_u
    
    res = minimize(
        lambda u: obj_and_grad(u)[0],
        start,
        jac=lambda u: obj_and_grad(u)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    c_res = res.x / np.linalg.norm(res.x)
    val, _ = _ratio_and_grad_spectral(c_res, Q, T)
    return c_res, val

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    Q, T = _precompute_tensors(X, w_flat)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    c0 = np.zeros(21); c0[10] = 1.0; starts.append(c0)
    c10 = np.zeros(21); c10[20] = 1.0; starts.append(c10)
    c_10 = np.zeros(21); c_10[0] = 1.0; starts.append(c_10)
    c5 = np.zeros(21); c5[15] = 1.0; starts.append(c5)
    c4 = np.zeros(21); c4[14] = 1.0; starts.append(c4)
    
    for m in [1, 2, 3, 6, 7, 8, 9]:
        cm = np.zeros(21); cm[10 + m] = 1.0; starts.append(cm)
        cmn = np.zeros(21); cmn[10 - m] = 1.0; starts.append(cmn)
    
    for _ in range(300):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
    
    for idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 5:
            break
        try:
            c_res, val = _sphere_projected_minimize(start, Q, T, maxiter=300)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c_res.copy()
        except Exception:
            pass

    if best_coeffs is not None and time.time() < _DEADLINE - 10:
        try:
            c_res, val = _sphere_projected_minimize(best_coeffs, Q, T, maxiter=2000)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = c_res.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
