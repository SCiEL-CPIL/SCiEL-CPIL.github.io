import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import time

def rudin_problem10() -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    l = 10
    N_theta = 70
    N_phi = 130
    
    # 1. Quadrature Points
    # Gauss-Legendre nodes and weights for integral from -1 to 1
    x_nodes, x_weights = sp.roots_legendre(N_theta)
    # x_nodes are cos(theta)
    # theta = arccos(x_nodes)
    
    # Equally spaced phi
    phi_nodes = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    phi_weights = np.full(N_phi, 2 * np.pi / N_phi)
    
    # Meshgrid
    # We want a flat array of points
    # x_grid: (N_theta, N_phi)
    # phi_grid: (N_theta, N_phi)
    # weights: (N_theta, N_phi)
    
    x_grid, phi_grid = np.meshgrid(x_nodes, phi_nodes, indexing='ij')
    w_grid = np.outer(x_weights, phi_weights)
    
    # Flatten
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    w_flat = w_grid.flatten()
    
    N_points = len(x_flat)
    
    # 2. Precompute Basis Matrix A (N_points x 21)
    # m ranges from -10 to 10
    m_vals = np.arange(-l, l + 1)
    A = np.zeros((N_points, 2 * l + 1))
    
    # Precompute factorials for normalization
    # N_m = sqrt( (2l+1)/(4*pi) * (l-m)! / (l+m)! )
    # We can compute this on the fly or precompute for m=0..10
    
    # Precompute associated Legendre polynomials P_l^m(x) for m=0..10
    # sp.lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase
    
    # We need P_l^m for m=0..10
    # x_flat is cos(theta)
    
    # Store P_l^m for each m
    P_vals = {}
    for m in range(l + 1):
        P_vals[m] = sp.lpmv(m, l, x_flat)
        
    # Normalization constants
    N_norm = {}
    for m in range(l + 1):
        # (l-m)! / (l+m)!
        # Use gammaln for stability
        log_fact_ratio = sp.gammaln(l - m + 1) - sp.gammaln(l + m + 1)
        N_norm[m] = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(log_fact_ratio))
    
    # Fill A
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        norm = N_norm[abs_m]
        P_m = P_vals[abs_m]
        
        if m == 0:
            A[:, idx] = norm * P_m
        elif m > 0:
            A[:, idx] = np.sqrt(2) * norm * P_m * np.cos(m * phi_flat)
        else:
            A[:, idx] = np.sqrt(2) * norm * P_m * np.sin(abs_m * phi_flat)
            
    # 3. Define Objective Function
    # Minimize R(c) = ( ||f||_4 / ||f||_2 )
    # ||f||_2^2 = sum(w * f^2)
    # ||f||_4^4 = sum(w * f^4)
    # R^4 = sum(w * f^4) / (sum(w * f^2))^2
    
    def objective(c):
        f = A @ c
        f2 = f * f
        f4 = f2 * f2
        S2 = np.dot(w_flat, f2)
        S4 = np.dot(w_flat, f4)
        
        if S2 < 1e-12:
            return 1e12
            
        R4 = S4 / (S2 * S2)
        # We want to minimize R, which is R4^(1/4)
        # Minimizing R4 is equivalent and avoids the 4th root in the inner loop
        return R4

    def objective_grad(c):
        f = A @ c
        f2 = f * f
        f4 = f2 * f2
        S2 = np.dot(w_flat, f2)
        S4 = np.dot(w_flat, f4)
        
        if S2 < 1e-12:
            return np.zeros_like(c)
            
        # R4 = S4 / S2^2
        # d(R4)/dc = (1/S2^2) d(S4)/dc - (2*S4/S2^3) d(S2)/dc
        # d(S2)/dc = 2 * A^T (w * f)
        # d(S4)/dc = 4 * A^T (w * f^3)
        
        dS2_dc = 2 * (A.T @ (w_flat * f))
        dS4_dc = 4 * (A.T @ (w_flat * f * f2))
        
        grad = (1.0 / (S2 * S2)) * dS4_dc - (2.0 * S4 / (S2 ** 3)) * dS2_dc
        return grad

    # 4. Optimization
    # We want to find c that minimizes objective(c).
    # The problem is scale invariant.
    # Let's use multiple random restarts.
    
    best_c = np.ones(2 * l + 1)
    best_score = objective(best_c)
    
    # Try the seed
    c_seed = np.ones(2 * l + 1)
    score_seed = objective(c_seed)
    if score_seed < best_score:
        best_score = score_seed
        best_c = c_seed.copy()
        
    # Random restarts
    np.random.seed(42)
    max_restarts = 50
    current_time = time.time()
    
    for i in range(max_restarts):
        if current_time > time_limit - safety_margin:
            break
            
        # Random initial guess
        # Normalize to unit norm to avoid scaling issues
        c0 = np.random.randn(2 * l + 1)
        c0 /= np.linalg.norm(c0)
        
        try:
            res = opt.minimize(objective, c0, jac=objective_grad, method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                c_opt = res.x
                # Normalize c_opt to ensure it's not degenerate
                norm_c = np.linalg.norm(c_opt)
                if norm_c > 1e-9:
                    c_opt /= norm_c
                    score_opt = objective(c_opt)
                    if score_opt < best_score:
                        best_score = score_opt
                        best_c = c_opt.copy()
        except Exception as e:
            pass
            
        # Also try a few structured initial guesses
        # e.g. only m=0, only m=10, etc.
        # But random restarts should cover the space.
        
        current_time = time.time()
        
    # Final polish with tighter tolerances on the best solution found
    if best_score < 1e10: # If we found a valid solution
        try:
            res_final = opt.minimize(objective, best_c, jac=objective_grad, method='L-BFGS-B', 
                                     options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res_final.success:
                c_final = res_final.x
                norm_c = np.linalg.norm(c_final)
                if norm_c > 1e-9:
                    c_final /= norm_c
                    score_final = objective(c_final)
                    if score_final < best_score:
                        best_score = score_final
                        best_c = c_final.copy()
        except Exception as e:
            pass

    # Return the coefficients
    # The problem asks for 21 real coefficients.
    # The validity check: max|c| < 1e-3 or max|c| > 1e5.
    # Our c is normalized to unit norm, so max|c| <= 1. This is safe.
    
    return best_c
