# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m=-10..10) for degree-10 spherical harmonics
    that minimize R = ||f||_4 / ||f||_2 on the unit sphere.
    
    Strategy: Multi-restart L-BFGS-B optimization with diverse initializations
    exploiting the rotational symmetry of the problem.
    """
    k = 10
    start_time = time.time()
    deadline = 600.0  # 600s wall clock limit
    
    # 1. Setup Quadrature
    n_x = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_x)
    dphi = 2.0 * np.pi / n_phi
    phi_vals = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)
    
    X, PHI = np.meshgrid(x, phi_vals, indexing='ij')
    W = np.outer(w, np.ones(n_phi) * dphi)
    
    N_pts = n_x * n_phi
    x_flat = X.ravel()
    phi_flat = PHI.ravel()
    W_flat = W.ravel()
    
    # 2. Precompute Basis Functions
    P_vals = np.zeros((11, N_pts))
    for m in range(11):
        P_vals[m] = lpmv(m, k, x_flat)
        
    cos_m_phi = np.zeros((10, N_pts))
    sin_m_phi = np.zeros((10, N_pts))
    for m in range(1, 11):
        cos_m_phi[m-1] = np.cos(m * phi_flat)
        sin_m_phi[m-1] = np.sin(m * phi_flat)
        
    N_m = np.zeros(11)
    for m in range(11):
        num = factorial(k - m, exact=False)
        den = factorial(k + m, exact=False)
        N_m[m] = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
    B = np.zeros((N_pts, 21))
    B[:, 10] = N_m[0] * P_vals[0]
    for m in range(1, 11):
        B[:, 10 + m] = np.sqrt(2.0) * N_m[m] * P_vals[m] * cos_m_phi[m-1]
        B[:, 10 - m] = np.sqrt(2.0) * N_m[m] * P_vals[m] * sin_m_phi[m-1]
    
    # Verify orthonormality (should be ~I)
    # B_T_W_B = B.T @ np.diag(W_flat) @ B  # Too expensive, skip
    
    # 3. Define objective function
    def objective(c):
        """Return R = ||f||_4 / ||f||_2"""
        f_vals = B @ c
        norm2_sq = np.sum(W_flat * f_vals**2)
        if norm2_sq < 1e-15:
            return 1e10
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sum(W_flat * f_vals**4) ** 0.25
        return norm4 / norm2
    
    def objective_with_grad(c):
        """Return R and its gradient"""
        f_vals = B @ c
        norm2_sq = np.sum(W_flat * f_vals**2)
        if norm2_sq < 1e-15:
            return 1e10, np.zeros(21)
        norm2 = np.sqrt(norm2_sq)
        norm4_4 = np.sum(W_flat * f_vals**4)
        norm4 = norm4_4 ** 0.25
        R = norm4 / norm2
        
        # Gradient of R = norm4/norm2
        # dR/dc = (1/norm2) * d(norm4)/dc - (norm4/norm2^2) * d(norm2)/dc
        # d(norm4_4)/dc = 4 * sum(W * f^3 * B)
        # d(norm2_sq)/dc = 2 * sum(W * f * B)
        # d(norm4)/dc = (1/4) * norm4_4^(-3/4) * d(norm4_4)/dc
        # d(norm2)/dc = (1/2) * norm2_sq^(-1/2) * d(norm2_sq)/dc
        
        df_dc = B  # (N_pts, 21)
        
        dnorm4_4_dc = 4.0 * (W_flat[:, None] * f_vals**3) @ df_dc
        dnorm2_sq_dc = 2.0 * (W_flat[:, None] * f_vals) @ df_dc
        
        if norm4_4 < 1e-15:
            dnorm4_dc = np.zeros(21)
        else:
            dnorm4_dc = 0.25 * (norm4_4 ** (-0.75)) * dnorm4_4_dc
        
        if norm2_sq < 1e-15:
            dnorm2_dc = np.zeros(21)
        else:
            dnorm2_dc = 0.5 * (norm2_sq ** (-0.5)) * dnorm2_sq_dc
        
        grad_R = (1.0 / norm2) * dnorm4_dc - (norm4 / (norm2**2)) * dnorm2_dc
        
        return R, grad_R
    
    # 4. Multi-restart optimization
    np.random.seed(42)
    
    best_R = np.inf
    best_c = np.ones(21)
    best_c /= np.linalg.norm(best_c)
    
    # Generate diverse initializations
    initializations = []
    
    # Start 1: Pure m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    initializations.append(c0)
    
    # Start 2: Pure m=10
    c10 = np.zeros(21)
    c10[20] = 1.0
    initializations.append(c10)
    
    # Start 3: Pure m=-10
    cm10 = np.zeros(21)
    cm10[0] = 1.0
    initializations.append(cm10)
    
    # Start 4: Equal mix of m=0 and m=10
    cmix1 = np.zeros(21)
    cmix1[10] = 1.0 / np.sqrt(2)
    cmix1[20] = 1.0 / np.sqrt(2)
    initializations.append(cmix1)
    
    # Start 5: Equal mix of m=0 and m=-10
    cmix2 = np.zeros(21)
    cmix2[10] = 1.0 / np.sqrt(2)
    cmix2[0] = 1.0 / np.sqrt(2)
    initializations.append(cmix2)
    
    # Random initializations
    for _ in range(20):
        c_rand = np.random.randn(21)
        c_rand /= np.linalg.norm(c_rand)
        initializations.append(c_rand)
    
    # Also try some structured random: only even m or only odd m
    for _ in range(10):
        c_even = np.zeros(21)
        for m in range(-10, 11, 2):
            c_even[m + 10] = np.random.randn()
        if np.linalg.norm(c_even) > 1e-10:
            c_even /= np.linalg.norm(c_even)
            initializations.append(c_even)
    
    for _ in range(10):
        c_odd = np.zeros(21)
        for m in range(-9, 10, 2):
            c_odd[m + 10] = np.random.randn()
        if np.linalg.norm(c_odd) > 1e-10:
            c_odd /= np.linalg.norm(c_odd)
            initializations.append(c_odd)
    
    max_restarts = len(initializations)
    
    for restart_idx, c_init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                objective_with_grad,
                c_init,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < best_R:
                R_val = result.fun
                c_val = result.x
                
                # Normalize
                norm_c = np.linalg.norm(c_val)
                if norm_c > 1e-15:
                    c_val = c_val / norm_c
                    R_val = objective(c_val)
                
                if R_val < best_R:
                    best_R = R_val
                    best_c = c_val.copy()
                    
        except Exception as e:
            continue
    
    # Final validation
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(21)
        
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3 or max_abs_c > 1e5:
        best_c /= max_abs_c
    
    return best_c

# EVOLVE-BLOCK-END
