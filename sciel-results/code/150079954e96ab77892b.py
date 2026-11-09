import numpy as np
from scipy.special import lpmv
from scipy.optimize import differential_evolution

def rudin_problem10() -> np.ndarray:
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 2 * k + 1
    
    # 1. Construct Grid and Weights
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = np.ones(n_phi) * (2 * np.pi / n_phi)
    
    cos_theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing='ij')
    cos_theta_flat = cos_theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    w = np.outer(w_theta, w_phi).flatten()
    N_points = n_theta * n_phi
    
    # 2. Construct Basis Matrix Y (N_points x 21)
    Y = np.zeros((N_points, dim))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        log_num = np.sum(np.log(np.arange(1, 11 - abs_m))) if abs_m < 10 else 0
        log_den = np.sum(np.log(np.arange(1, 11 + abs_m)))
        ratio = np.exp(log_num - log_den)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
        
        P = lpmv(abs_m, k, cos_theta_flat)
        
        idx = m + k
        if m == 0:
            Y[:, idx] = N_m * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.cos(m * phi_flat)
        else:
            Y[:, idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi_flat)
    
    # Precompute weighted Y for potential gradient usage later if needed
    WY = Y * w[:, None]

    def compute_R(c):
        """Compute R = ||f||_4 / ||f||_2"""
        f = Y @ c
        f2 = f * f
        norm2_sq = np.sum(w * f2)
        norm4_sq = np.sum(w * f2 * f2)
        if norm2_sq < 1e-15:
            return 1e6
        return np.sqrt(norm4_sq / norm2_sq)

    def objective(c_raw):
        """
        Objective for Differential Evolution.
        c_raw is in [-1, 1]^dim. We normalize it to the sphere.
        """
        norm = np.linalg.norm(c_raw)
        if norm < 1e-8:
            return 1e6
        c = c_raw / norm
        return compute_R(c)

    # 3. Global Optimization using Differential Evolution
    # bounds for each coefficient
    bounds = [(-1.0, 1.0)] * dim
    
    # Run DE
    # popsize=20, maxiter=500 is a reasonable start. 
    # tol=1e-8 for convergence.
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=1000, 
        popsize=25, 
        tol=1e-8,
        seed=42,
        polish=True # Polish with L-BFGS-B at the end
    )
    
    best_c_raw = result.x
    best_R = result.fun
    
    # Normalize best_c_raw to unit sphere
    best_c = best_c_raw / np.linalg.norm(best_c_raw)
    
    # 4. Optional: Final polish with custom gradient descent on sphere 
    # to ensure high precision, as DE/L-BFGS might stop at tol.
    # We will do a few steps of projected gradient descent.
    
    def compute_grad_quartic(c):
        f = Y @ c
        f3 = f * f * f
        grad = 4.0 * (WY.T @ f3)
        return grad

    # Gradient Descent Polish
    step_size = 0.01
    for _ in range(100):
        grad = compute_grad_quartic(best_c)
        grad_tan = grad - np.dot(grad, best_c) * best_c
        
        # Line search
        alpha = step_size
        best_alpha = 0
        best_R_curr = compute_R(best_c)
        
        for _ in range(10):
            c_new = best_c - alpha * grad_tan
            c_new /= np.linalg.norm(c_new)
            R_new = compute_R(c_new)
            if R_new < best_R_curr:
                best_R_curr = R_new
                best_c = c_new
                best_alpha = alpha
            else:
                alpha *= 0.5
                if alpha < 1e-6:
                    break
        
        if best_alpha == 0:
            break

    # Validation
    if not np.all(np.isfinite(best_c)) or np.max(np.abs(best_c)) < 1e-3 or np.max(np.abs(best_c)) > 1e5:
        # Fallback to a valid solution
        best_c = np.zeros(dim)
        best_c[k] = 1.0

    return best_c
