import numpy as np
from scipy.special import lpmv
import time
import math

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    k = 10
    n_theta = 70
    n_phi = 130
    n_points = n_theta * n_phi
    
    # 1. Precompute quadrature nodes and weights
    mu, w_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    w_theta_rep = np.repeat(w_theta, n_phi)
    w_flat = w_theta_rep * w_phi
    
    # 2. Precompute Basis Functions Y_m
    m_vals = np.arange(-k, k + 1)
    Y_basis = np.zeros((n_points, 2 * k + 1), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        P_lm = lpmv(abs_m, k, mu_flat)
        
        fact_num = math.factorial(k - abs_m)
        fact_den = math.factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * fact_num / fact_den)
        
        if m == 0:
            Y_basis[:, idx] = N_m * P_lm
        elif m > 0:
            Y_basis[:, idx] = np.sqrt(2) * N_m * P_lm * np.cos(m * phi_flat)
        else:
            Y_basis[:, idx] = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * phi_flat)
    
    # Precompute Y^T W Y for efficiency (21x21 matrix)
    # Actually, we need to compute f = Y @ c, then f^2, f^4 weighted sums
    # Let's precompute Y^T W for fast matrix-vector products
    YTW = Y_basis.T @ np.diag(w_flat)  # 21 x n_points @ n_points x 21 -> too slow
    # Better: just use Y_basis.T @ (w_flat * something)
    
    def compute_ratio_and_riemannian_grad(c):
        """Compute J = ||f||_4^4 / ||f||_2^4 and Riemannian gradient on unit sphere."""
        f = Y_basis @ c
        f2 = f * f
        f4 = f2 * f2
        
        Q2 = np.dot(w_flat, f2)
        Q4 = np.dot(w_flat, f4)
        
        if Q2 < 1e-12:
            return 1e10, np.zeros_like(c)
        
        J = Q4 / (Q2 * Q2)
        
        # Euclidean gradient of J
        w_f3 = w_flat * (f * f2)  # w * f^3
        w_f = w_flat * f
        
        grad_Q4 = 4.0 * (Y_basis.T @ w_f3)
        grad_Q2 = 2.0 * (Y_basis.T @ w_f)
        
        grad_E = (grad_Q4 / (Q2 * Q2)) - (2.0 * Q4 / (Q2 ** 3)) * grad_Q2
        
        # Riemannian gradient: project onto tangent space of unit sphere
        # Tangent space at c: {v : v · c = 0}
        # Projection: grad_R = grad_E - (grad_E · c) * c
        dot_gc = np.dot(grad_E, c)
        grad_R = grad_E - dot_gc * c
        
        return J, grad_R
    
    def riemannian_exponential(c, v):
        """Exponential map on the unit sphere."""
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            return c.copy()
        return c * np.cos(norm_v) + (v / norm_v) * np.sin(norm_v)
    
    def riemannian_line_search(c, grad_R, J_current, max_iter=50):
        """Backtracking line search on the manifold."""
        step = 0.1
        for _ in range(max_iter):
            v = -step * grad_R
            c_new = riemannian_exponential(c, v)
            J_new, _ = compute_ratio_and_riemannian_grad(c_new)
            if J_new < J_current - 1e-4 * step * np.dot(grad_R, grad_R):
                return c_new, J_new, step
            step *= 0.5
        return c, J_current, step
    
    best_c = np.ones(2 * k + 1)
    best_c /= np.linalg.norm(best_c)
    best_val, _ = compute_ratio_and_riemannian_grad(best_c)
    
    np.random.seed(42)
    
    # Multiple restarts with Riemannian optimization
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() > deadline - 100:
            break
        
        # Generate initial point on unit sphere
        if i == 0:
            c0 = np.ones(2 * k + 1)
        else:
            c0 = np.random.randn(2 * k + 1)
        c0 /= np.linalg.norm(c0)
        
        # Riemannian gradient descent with adaptive step size
        c_curr = c0.copy()
        J_curr, grad_R = compute_ratio_and_riemannian_grad(c_curr)
        
        # Adaptive step size based on initial gradient magnitude
        grad_norm = np.linalg.norm(grad_R)
        if grad_norm < 1e-10:
            continue
        
        step_size = 0.1
        max_iter = 500
        tol = 1e-12
        
        for it in range(max_iter):
            if time.time() > deadline - 80:
                break
            
            # Line search
            c_new, J_new, step_used = riemannian_line_search(c_curr, grad_R, J_curr)
            
            if J_new >= J_curr - 1e-15:
                # Try smaller step or break
                if step_used < 1e-8:
                    break
                continue
            
            c_curr = c_new
            J_curr = J_new
            
            # Check convergence
            _, grad_R_new = compute_ratio_and_riemannian_grad(c_curr)
            if np.linalg.norm(grad_R_new) < tol:
                break
            
            # Adapt step size
            if J_new < J_curr:
                step_size = min(step_size * 1.2, 1.0)
            else:
                step_size *= 0.8
            
            grad_R = grad_R_new
        
        if J_curr < best_val:
            best_val = J_curr
            best_c = c_curr.copy()
    
    # High-precision polish using Riemannian optimization with tighter tolerances
    if time.time() < deadline - 30:
        c_curr = best_c.copy()
        J_curr, grad_R = compute_ratio_and_riemannian_grad(c_curr)
        
        max_iter_polish = 1000
        tol_polish = 1e-14
        
        for it in range(max_iter_polish):
            if time.time() > deadline - 10:
                break
            
            c_new, J_new, _ = riemannian_line_search(c_curr, grad_R, J_curr, max_iter=30)
            
            if J_new >= J_curr - 1e-16:
                break
            
            c_curr = c_new
            J_curr = J_new
            
            _, grad_R_new = compute_ratio_and_riemannian_grad(c_curr)
            if np.linalg.norm(grad_R_new) < tol_polish:
                break
            
            grad_R = grad_R_new
        
        if J_curr < best_val:
            best_val = J_curr
            best_c = c_curr.copy()
    
    # Ensure validity: max|c| between 1e-3 and 1e5
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c *= 1.0 / max_abs
    elif max_abs > 1e5:
        best_c *= 1e5 / max_abs
        
    return best_c
