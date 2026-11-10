# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from math import lgamma, pi as math_pi

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m=-10..10) for degree-10 spherical harmonics
    that minimize R = ||f||_4 / ||f||_2 over the unit sphere S^2.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    # Deadline guard: 630s wall-clock limit, keep safety margin
    deadline = time.time() + 600 - 45  # 45s safety margin
    
    # --- Precompute quadrature and basis ---
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    ct = x_gl
    st = np.sqrt(np.maximum(0, 1 - ct**2))
    
    # Equally spaced phi
    phi = np.linspace(0, 2 * math_pi, n_phi, endpoint=False)
    dphi = 2 * math_pi / n_phi
    
    # Precompute associated Legendre polynomials P_10^m(cos theta) for m=0..10
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, ct)
    
    # Precompute cos(m*phi) and sin(m*phi) for m = 1..10
    cos_m_phi = np.zeros((n_phi, k))
    sin_m_phi = np.zeros((n_phi, k))
    for m in range(1, k + 1):
        cos_m_phi[:, m-1] = np.cos(m * phi)
        sin_m_phi[:, m-1] = np.sin(m * phi)
    
    # Build the full basis using Kronecker products
    n_pts = n_theta * n_phi
    B = np.zeros((n_pts, n_coeffs))
    
    def compute_N(m):
        abs_m = abs(m)
        log_N_sq = np.log(21.0 / (4.0 * math_pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
        return np.sqrt(np.exp(log_N_sq))
    
    N_0 = compute_N(0)
    val_theta_0 = N_0 * P[:, 0]
    B[:, k] = np.tile(val_theta_0, n_phi)
    
    for m in range(1, k + 1):
        N_m = compute_N(m)
        val_theta = np.sqrt(2.0) * N_m * P[:, m]
        B[:, k + m] = np.kron(val_theta, cos_m_phi[:, m-1])
        B[:, k - m] = np.kron(val_theta, sin_m_phi[:, m-1])
    
    # Quadrature weights
    w = np.repeat(w_gl, n_phi) * dphi
    
    # --- Objective: minimize R^4 = S4 / S2^2 ---
    def objective(c):
        f = B @ c
        f2 = f * f
        f4 = f2 * f2
        S2 = np.sum(w * f2)
        S4 = np.sum(w * f4)
        if S2 <= 0:
            return np.inf
        return S4 / (S2 * S2)
    
    def gradient(c):
        f = B @ c
        f2 = f * f
        S2 = np.sum(w * f2)
        S4 = np.sum(w * f4)
        if S2 <= 0:
            return np.zeros_like(c)
        grad_S4 = 4.0 * (B.T @ (w * f**3))
        grad_S2 = 2.0 * (B.T @ (w * f))
        return (grad_S4 * S2 - 2.0 * S4 * grad_S2) / (S2 * S2 * S2)
    
    # --- Conjugate gradient on the sphere with projection ---
    def cg_sphere(c0, max_iter=500, tol=1e-14):
        """
        Conjugate gradient method on the unit sphere.
        Uses a Riemannian CG approach: at each step, project gradient to tangent space,
        take a step, then re-project to the sphere.
        """
        c = c0 / np.linalg.norm(c0)
        n = len(c)
        
        g = gradient(c)
        # Project gradient to tangent space
        g = g - np.dot(g, c) * c
        
        # Initial search direction
        d = -g
        d = d - np.dot(d, c) * c
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Line search: find alpha that minimizes objective along the geodesic
            # c(alpha) = c * cos(alpha) + d * sin(alpha) / ||d||
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                break
            
            # Try a few step sizes
            best_alpha = 0.01
            best_val = objective(c)
            
            for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
                c_new = c * np.cos(alpha) + d * np.sin(alpha) / d_norm
                c_new = c_new / np.linalg.norm(c_new)
                val = objective(c_new)
                if val < best_val:
                    best_val = val
                    best_alpha = alpha
            
            # Update
            c_new = c * np.cos(best_alpha) + d * np.sin(best_alpha) / d_norm
            c_new = c_new / np.linalg.norm(c_new)
            
            g_new = gradient(c_new)
            g_new = g_new - np.dot(g_new, c_new) * c_new
            
            # Check convergence
            if np.linalg.norm(g_new) < tol:
                c = c_new
                break
            
            # Polak-Ribiere beta
            beta = np.dot(g_new, g_new - g) / (np.dot(g, g) + 1e-30)
            if beta < 0:
                beta = 0
            
            # New search direction
            d = -g_new + beta * d
            # Re-orthogonalize against c
            d = d - np.dot(d, c) * c
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                d = -g_new
                d = d - np.dot(d, c) * c
            
            c = c_new
            g = g_new
        
        return c, objective(c)
    
    # --- L-BFGS-B on sphere as fallback ---
    from scipy.optimize import minimize
    
    def sphere_objective(c):
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.inf
        c_norm = c / norm
        return objective(c_norm)
    
    def sphere_grad(c):
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.zeros_like(c)
        c_norm = c / norm
        g = gradient(c_norm)
        return g - np.dot(g, c_norm) * c_norm
    
    # Generate diverse initial points
    rng = np.random.default_rng(12345)
    
    best_val = np.inf
    best_c = None
    
    n_restarts = 80
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial point
        if i == 0:
            c0 = np.ones(n_coeffs) / np.sqrt(n_coeffs)
        elif i == 1:
            c0 = np.ones(n_coeffs)
            c0[::2] = -1.0
            c0 /= np.linalg.norm(c0)
        elif i == 2:
            c0 = np.zeros(n_coeffs)
            for j in range(0, n_coeffs, 2):
                c0[j] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 3:
            c0 = np.zeros(n_coeffs)
            for j in range(1, n_coeffs, 2):
                c0[j] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 4:
            c0 = np.zeros(n_coeffs)
            c0[k] = 1.0
        elif i == 5:
            # Symmetric: only m=0 and m=±10
            c0 = np.zeros(n_coeffs)
            c0[k] = 1.0
            c0[0] = 1.0
            c0[-1] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 6:
            # Only |m| = 10
            c0 = np.zeros(n_coeffs)
            c0[0] = 1.0
            c0[-1] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 7:
            # Only |m| = 8
            c0 = np.zeros(n_coeffs)
            c0[k-8] = 1.0
            c0[k+8] = 1.0
            c0 /= np.linalg.norm(c0)
        else:
            c0 = rng.standard_normal(n_coeffs)
            c0 /= np.linalg.norm(c0)
        
        # Use CG method
        try:
            c_res, val_res = cg_sphere(c0, max_iter=300, tol=1e-13)
            if val_res < best_val:
                best_val = val_res
                best_c = c_res.copy()
        except Exception:
            pass
        
        # Also try L-BFGS-B if time permits
        if time.time() < deadline - 30:
            try:
                res = minimize(
                    sphere_objective,
                    c_res if best_c is not None else c0,
                    jac=sphere_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                c_lbfgs = res.x
                norm = np.linalg.norm(c_lbfgs)
                if norm > 1e-15:
                    c_lbfgs /= norm
                val_lbfgs = objective(c_lbfgs)
                if val_lbfgs < best_val:
                    best_val = val_lbfgs
                    best_c = c_lbfgs.copy()
            except Exception:
                pass
    
    # Final polish: run CG with tighter tolerance on the best solution
    if best_c is not None and time.time() < deadline - 10:
        try:
            c_polished, val_polished = cg_sphere(best_c, max_iter=500, tol=1e-15)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass
    
    # If no valid result, fall back
    if best_c is None:
        best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Scale to valid range
        if max_abs < 1e-3:
            best_c *= 1e-3 / max_abs
        else:
            best_c *= 1e5 / max_abs
    
    return best_c

# EVOLVE-BLOCK-END
