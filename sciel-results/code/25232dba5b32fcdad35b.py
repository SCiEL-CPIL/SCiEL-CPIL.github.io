# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_{m=-10}^{10} c_m Y_m that minimizes the L4/L2 norm ratio.
    
    Returns:
        coeffs: np.ndarray of shape (21,) with real coefficients c_m ordered m = -10..10
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 590s deadline with safety margin
    
    k = 10
    
    # Precompute quadrature points and weights
    n_theta = 70
    n_phi = 130
    
    x_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    u_grid, phi_grid = np.meshgrid(x_theta, phi, indexing='ij')
    
    w_phi = 2*np.pi / n_phi
    weights = np.outer(w_theta, np.ones(n_phi)) * w_phi
    weights_flat = weights.ravel()
    
    # Precompute the spherical harmonic basis values at all quadrature points
    def N_coeff(m):
        m_abs = abs(m)
        return np.sqrt(21.0/(4*np.pi) * float(factorial(k - m_abs, exact=True)) / float(factorial(k + m_abs, exact=True)))
    
    n_points = n_theta * n_phi
    B = np.zeros((n_points, 2*k+1), dtype=np.float64)
    
    for m in range(-k, k+1):
        m_idx = m + k
        m_abs = abs(m)
        P = lpmv(m_abs, k, u_grid)
        N_m = N_coeff(m)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * phi_grid)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(m_abs * phi_grid)
        
        B[:, m_idx] = Y.ravel()
    
    # Precompute B^T for repeated use
    BT = B.T  # shape (21, n_points)
    
    def objective_and_grad(c):
        f = B @ c
        f2 = f**2
        f4 = f**4
        sum_w_f2 = np.dot(weights_flat, f2)
        sum_w_f4 = np.dot(weights_flat, f4)
        
        if sum_w_f2 <= 0:
            return 0.0, np.zeros_like(c)
        
        val = sum_w_f4 / (sum_w_f2**2)
        
        w_f3 = weights_flat * f**3
        w_f = weights_flat * f
        
        grad_sum_w_f4 = 4.0 * (BT @ w_f3)
        grad_sum_w_f2 = 2.0 * (BT @ w_f)
        
        grad = grad_sum_w_f4 / (sum_w_f2**2) - 2.0 * sum_w_f4 / (sum_w_f2**3) * grad_sum_w_f2
        
        return val, grad
    
    def power_iteration(c, n_iters=20):
        """Fixed-point iteration: c_new ∝ B^T(W*(B@c)^3), normalized to unit L2 norm of f."""
        for _ in range(n_iters):
            f = B @ c
            f3 = f**3
            w_f3 = weights_flat * f3
            c_new = BT @ w_f3
            # Normalize so that ||f||_2^2 = 1
            f_new = B @ c_new
            norm2 = np.dot(weights_flat, f_new**2)
            if norm2 > 0:
                c_new = c_new / np.sqrt(norm2)
            c = c_new
        return c
    
    def run_lbfgsb(c0, maxiter=300, ftol=1e-15, gtol=1e-12):
        try:
            result = minimize(
                objective_and_grad,
                c0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            return result.x, result.fun
        except Exception:
            return c0, np.inf
    
    best_val = np.inf
    best_c = np.ones(2*k+1, dtype=np.float64)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    n_restarts = 60
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart < 5:
            # Structured initializations: single dominant mode
            c0 = np.zeros(2*k+1)
            m_idx = rng.integers(0, 2*k+1)
            c0[m_idx] = 1.0
            # Add small random perturbation to break symmetry
            c0 += 0.01 * rng.standard_normal(2*k+1)
        elif restart < 15:
            # Two-mode initializations
            c0 = np.zeros(2*k+1)
            idx1, idx2 = rng.choice(2*k+1, size=2, replace=False)
            c0[idx1] = rng.standard_normal()
            c0[idx2] = rng.standard_normal()
            c0 += 0.01 * rng.standard_normal(2*k+1)
        else:
            # Random initializations with varying concentration
            concentration = rng.uniform(0.5, 3.0)
            c0 = rng.standard_normal(2*k+1) * concentration
            # Occasionally use sparse initializations
            if rng.random() < 0.3:
                mask = rng.random(2*k+1) < 0.4
                c0[~mask] = 0.0
        
        # Normalize to unit L2 norm of f
        f0 = B @ c0
        sum_w_f2 = np.dot(weights_flat, f0**2)
        if sum_w_f2 > 0:
            c0 = c0 / np.sqrt(sum_w_f2)
        
        # Power iteration warmup
        c_warm = power_iteration(c0, n_iters=15)
        
        # L-BFGS-B from warm-started point
        c_opt, val = run_lbfgsb(c_warm, maxiter=300, ftol=1e-15, gtol=1e-12)
        
        if val < best_val:
            best_val = val
            best_c = c_opt.copy()
    
    # Additional refinement with more iterations from the best point found
    if time.time() < deadline - 120:
        # Tighter polish
        c_polish, val = run_lbfgsb(best_c, maxiter=1000, ftol=1e-16, gtol=1e-14)
        if val < best_val:
            best_val = val
            best_c = c_polish.copy()
    
    # One more round of diverse restarts if time permits
    if time.time() < deadline - 90:
        for restart2 in range(20):
            if time.time() > deadline - 60:
                break
            c0 = rng.standard_normal(2*k+1)
            f0 = B @ c0
            sum_w_f2 = np.dot(weights_flat, f0**2)
            if sum_w_f2 > 0:
                c0 = c0 / np.sqrt(sum_w_f2)
            
            c_warm = power_iteration(c0, n_iters=10)
            c_opt, val = run_lbfgsb(c_warm, maxiter=200, ftol=1e-15, gtol=1e-12)
            
            if val < best_val:
                best_val = val
                best_c = c_opt.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 30:
        c_final, val = run_lbfgsb(best_c, maxiter=2000, ftol=1e-17, gtol=1e-15)
        if val < best_val:
            best_val = val
            best_c = c_final.copy()
    
    # Normalize the result to have reasonable magnitude
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1.0 / max_abs_c)
    
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(2*k+1, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
