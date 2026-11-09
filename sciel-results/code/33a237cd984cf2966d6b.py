import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    k = 10
    n_theta = 70
    n_phi = 130
    
    # 1. Construct Grid and Weights
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = np.ones(n_phi) * (2 * np.pi / n_phi)
    
    cos_theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing='ij')
    cos_theta_flat = cos_theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Total weight for each point
    w = np.outer(w_theta, w_phi).flatten()
    N_points = n_theta * n_phi
    
    # 2. Construct Basis Matrix Y (N_points x 21)
    # Y[:, m+10] corresponds to coefficient c[m+10]
    Y = np.zeros((N_points, 2 * k + 1))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # Calculate N_m = sqrt(21/(4*pi) * (10-m)! / (10+m)!)
        # Using logs for stability, though 10 is small
        log_num = np.sum(np.log(np.arange(1, 11 - abs_m))) if abs_m < 10 else 0
        log_den = np.sum(np.log(np.arange(1, 11 + abs_m)))
        # Note: (10-m)! is 0! = 1 if m=10, so sum is empty -> 0. Correct.
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
            
    # Precompute weighted Y to speed up dot products?
    # f = Y @ c
    # f^2 = (Y @ c)^2
    # norm2_sq = sum(w * f^2)
    # norm4_sq = sum(w * f^4)
    
    # 3. Global Random Search
    # We want to minimize R = sqrt(norm4_sq / norm2_sq)
    # This is equivalent to minimizing norm4_sq / norm2_sq.
    
    rng = np.random.default_rng(123)
    n_random = 50000
    dim = 2 * k + 1
    
    # Generate random vectors on the sphere
    C_rand = rng.normal(size=(n_random, dim))
    C_rand /= np.linalg.norm(C_rand, axis=1, keepdims=True)
    
    # Compute f for all random candidates
    # F has shape (n_random, N_points)
    # F = C_rand @ Y.T
    F = C_rand @ Y.T
    
    # Compute norms
    # f2 = F^2
    # f4 = f2^2
    F2 = F * F
    F4 = F2 * F2
    
    # norm2_sq[i] = sum(w * F2[i, :])
    # norm4_sq[i] = sum(w * F4[i, :])
    
    norm2_sq = F2 @ w
    norm4_sq = F4 @ w
    
    # R = sqrt(norm4_sq / norm2_sq)
    # To avoid division by zero (should not happen for unit vectors on sphere with non-zero basis),
    # we can just minimize norm4_sq / norm2_sq.
    # Since norm2_sq should be approx 1 (orthonormal basis), minimizing norm4_sq is a good proxy,
    # but let's stick to the ratio for precision.
    R_vals = np.sqrt(norm4_sq / norm2_sq)
    
    # Find top K candidates
    K = 20
    top_indices = np.argsort(R_vals)[:K]
    best_c = C_rand[top_indices[0]].copy()
    best_R = R_vals[top_indices[0]]
    
    # 4. Local Refinement
    # Use L-BFGS-B with equality constraint ||c||^2 = 1
    # Objective: R(c)
    
    def objective(c):
        f = Y @ c
        f2 = f * f
        norm2_sq = np.sum(w * f2)
        norm4_sq = np.sum(w * f2 * f2)
        return np.sqrt(norm4_sq / norm2_sq)
        
    def constraint_eq(c):
        return np.sum(c**2) - 1.0
        
    # Gradient for L-BFGS-B? 
    # L-BFGS-B can use numerical gradients, but analytical is better.
    # Let's try L-BFGS-B with numerical gradient first for simplicity, 
    # as dimension is small (21).
    
    for idx in top_indices:
        c_init = C_rand[idx].copy()
        try:
            res = minimize(
                objective, 
                c_init, 
                method='L-BFGS-B', 
                constraints=[{'type': 'eq', 'fun': constraint_eq}],
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.success:
                c_opt = res.x
                c_opt /= np.linalg.norm(c_opt)
                R_opt = objective(c_opt)
                if R_opt < best_R:
                    best_R = R_opt
                    best_c = c_opt
        except Exception:
            pass
            
    # Final check
    if best_R == np.inf:
        best_c = np.zeros(dim)
        best_c[k] = 1.0 # Fallback
        
    return best_c
