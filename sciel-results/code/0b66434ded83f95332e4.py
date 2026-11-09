import numpy as np
from scipy.special import lpmv
import math

def _compute_y_values():
    """
    Precompute the values of the 21 real orthonormal spherical harmonics
    for degree l=10 at the quadrature points.
    """
    l = 10
    N_theta = 70
    N_phi = 130
    
    x, w_gl = np.polynomial.legendre.leggauss(N_theta)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    theta = np.arccos(x)
    Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
    
    weights = np.outer(np.ones(N_phi), w_gl) * (2 * np.pi / N_phi)
    
    Y = np.zeros((N_phi, N_theta, 21), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + math.lgamma(11 - abs_m) - math.lgamma(11 + abs_m))
        N_m = np.exp(log_N)
        
        P_lm = lpmv(abs_m, l, x)
        
        if m == 0:
            Y[:, :, m_idx] = N_m * P_lm
        elif m > 0:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y[:, :, m_idx] = np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * Phi)
            
    return Y, weights

# Precompute
Y_vals, weights = _compute_y_values()
Y_flat = Y_vals.flatten() # (N_points, 21)
w_flat = weights.flatten() # (N_points,)

def _objective_and_grad(c):
    """
    Compute objective J(c) = sum w * f^4 and gradient.
    """
    f = Y_flat @ c
    f4 = f ** 4
    J = np.sum(w_flat * f4)
    
    wf3 = w_flat * (f ** 3)
    g = 4.0 * (Y_flat.T @ wf3)
    
    return J, g

def _riemannian_cg(c0, max_iter=500, tol=1e-12):
    """
    Riemannian Conjugate Gradient on the unit sphere.
    """
    c = c0 / np.linalg.norm(c0)
    
    J, g_euclid = _objective_and_grad(c)
    g = g_euclid - np.dot(g_euclid, c) * c
    
    p = -g
    beta = 0.0
    prev_g_norm_sq = np.dot(g, g)
    
    for k in range(max_iter):
        # Backtracking line search
        alpha = 1.0
        phi_0 = J
        slope = -prev_g_norm_sq
        
        for _ in range(50):
            c_new = c + alpha * p
            norm_c_new = np.linalg.norm(c_new)
            if norm_c_new < 1e-12:
                alpha *= 0.5
                continue
            c_new /= norm_c_new
            
            J_new, _ = _objective_and_grad(c_new)
            
            if J_new <= phi_0 + 1e-4 * alpha * slope:
                break
            alpha *= 0.5
        
        c = c_new
        J = J_new
        
        _, g_euclid_new = _objective_and_grad(c)
        g_new = g_euclid_new - np.dot(g_euclid_new, c) * c
        g_new_norm_sq = np.dot(g_new, g_new)
        
        if np.sqrt(g_new_norm_sq) < tol:
            break
            
        if prev_g_norm_sq > 1e-16:
            beta = g_new_norm_sq / prev_g_norm_sq
        else:
            beta = 0.0
            
        p_old_tangent = p - np.dot(p, c) * c
        p = -g_new + beta * p_old_tangent
        
        prev_g_norm_sq = g_new_norm_sq
        
    return c, J

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    n_coeffs = 21
    best_c = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    n_restarts = 150
    
    for i in range(n_restarts):
        c0 = rng.normal(size=n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        try:
            c_res, val_res = _riemannian_cg(c0, max_iter=500, tol=1e-12)
            
            if val_res < best_val:
                best_val = val_res
                best_c = c_res
        except Exception:
            pass
            
    if best_c is None:
        return np.ones(n_coeffs, dtype=np.float64)
        
    return best_c.astype(np.float64)

# EVOLVE-BLOCK-END
