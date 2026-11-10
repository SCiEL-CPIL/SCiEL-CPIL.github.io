import numpy as np
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix on the specified quadrature grid.
    2. Use fixed-point iteration based on Euler-Lagrange condition: f^3 projected back to degree-10.
    3. Combine with gradient-based polishing for final precision.
    4. Multiple diverse starting points to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)
    
    # m=0
    idx = 10
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis for fast operations
    B = A * w_flat[:, None]  # B[i,j] = w_i * A[i,j]
    # For projection: c_proj = A^T (w * f^3) = B^T @ (f^3)
    
    # 3. Objective and Gradient
    def objective_and_grad(c):
        """Compute R(c) = ||f||_4/||f||_2 and gradient. c need not be unit norm."""
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradient of R w.r.t. c (not constrained to sphere)
        # dR/dc = (1/N2) * dN4/dc - (N4/N2^2) * dN2/dc
        # dN4/dc = (1/4) N4^(-3) * 4 * d_phi * sum(w * f^3 * Y_m) = N4^(-3) * d_phi * sum(w*f^3*Y)
        # dN2/dc = (1/(2*N2)) * 2 * d_phi * sum(w * f * Y_m) = (1/N2) * d_phi * sum(w*f*Y)
        
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        sum_wf_A = d_phi * (B.T @ f_vals)
        
        dN4_dc = (N4 ** (-3)) * sum_wf3_A
        dN2_dc = (1.0 / N2) * sum_wf_A
        
        grad = dN4_dc / N2 - (N4 / (N2 ** 2)) * dN2_dc
        
        return R, grad

    def objective_unit(u):
        """Compute R for unit vector u."""
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        if norm2_sq < 1e-12:
            return 1e10
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        return N4 / N2

    def grad_tangent(u):
        """Gradient projected to tangent space of unit sphere."""
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        if norm2_sq < 1e-12:
            return np.zeros(n_m)
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        sum_wf_A = d_phi * (B.T @ f_vals)
        
        dN4_dc = (N4 ** (-3)) * sum_wf3_A
        dN2_dc = (1.0 / N2) * sum_wf_A
        
        grad = dN4_dc / N2 - (N4 / (N2 ** 2)) * dN2_dc
        
        # Project to tangent
        u_dot_grad = np.dot(u, grad)
        return grad - u * u_dot_grad

    # 4. Fixed-point iteration based on Euler-Lagrange condition
    def fixed_point_iter(u, max_iter=500, omega=0.5, tol=1e-12):
        """
        Fixed-point iteration: project f^3 onto degree-10 harmonics.
        The Euler-Lagrange condition says f^3 should be proportional to a degree-10 harmonic.
        """
        u = u.copy()
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            return u
        u = u / u_norm
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            f_vals = A @ u
            f3 = f_vals ** 3
            
            # Project f^3 onto degree-10 basis
            c_proj = B.T @ f3  # This is A^T (w * f^3)
            
            c_proj_norm = np.linalg.norm(c_proj)
            if c_proj_norm < 1e-12:
                break
            
            c_proj = c_proj / c_proj_norm
            
            # Over-relaxation
            u_new = (1 - omega) * u + omega * c_proj
            u_new_norm = np.linalg.norm(u_new)
            if u_new_norm < 1e-12:
                break
            u_new = u_new / u_new_norm
            
            # Check convergence
            change = np.linalg.norm(u_new - u)
            u = u_new
            if change < tol:
                break
        
        return u

    # 5. Gradient polish on unit sphere
    def sphere_polish(u, max_iter=3000, lr_init=0.01, tol=1e-14):
        """Polish on the unit sphere using projected gradient with backtracking."""
        u = u.copy()
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            return u
        u = u / u_norm
        
        R_curr = objective_unit(u)
        if not np.isfinite(R_curr):
            return u
        
        grad_t = grad_tangent(u)
        d = -grad_t
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            return u
        
        beta = 0.0
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Backtracking line search with Armijo condition
            step = lr
            c1 = 1e-4
            success = False
            for _ in range(40):
                u_new = u + step * d
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                R_new = objective_unit(u_new)
                
                # Armijo condition
                if R_new < R_curr + c1 * step * np.dot(grad_t, d):
                    u = u_new
                    R_curr = R_new
                    success = True
                    break
                else:
                    step *= 0.5
            
            if not success:
                # Try even smaller steps
                for trial_step in [lr * 0.01, lr * 0.001, lr * 0.0001]:
                    u_new = u + trial_step * d
                    u_new_norm = np.linalg.norm(u_new)
                    if u_new_norm < 1e-12:
                        continue
                    u_new = u_new / u_new_norm
                    R_new = objective_unit(u_new)
                    if R_new < R_curr:
                        u = u_new
                        R_curr = R_new
                        success = True
                        break
                if not success:
                    break
            
            # Update gradient
            grad_t_new = grad_tangent(u)
            grad_norm = np.linalg.norm(grad_t_new)
            if grad_norm < tol:
                break
            
            # Polak-Ribiere beta
            d_old = d
            d_new = -grad_t_new
            
            if it > 0:
                num = np.dot(grad_t_new, grad_t_new - grad_t)
                den = np.dot(grad_t, grad_t)
                if den > 1e-14:
                    beta = max(0, num / den)
                else:
                    beta = 0
            
            d = d_new + beta * d_old if it > 0 else d_new
            d = d - u * np.dot(u, d)  # Re-project to tangent space
            
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-14:
                break
            
            # Adaptive step size
            if it % 50 == 0 and it > 0:
                lr = min(lr * 1.3, 0.05)
            elif it % 50 == 0:
                lr = max(lr * 0.8, 0.0005)
        
        return u

    # 6. Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(12345)
    
    # Pure modes
    for m in range(0, 11):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)
    
    for m in range(1, 11):
        c = np.zeros(n_m)
        c[10 - m] = 1.0
        candidates.append(c)
    
    # Pairs of cos modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c = np.zeros(n_m)
            c[10 + m1] = 1.0
            c[10 + m2] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Mix cos and sin for same |m|
    for m in range(1, 11):
        for angle in np.linspace(0, np.pi/2, 7):
            c = np.zeros(n_m)
            c[10 + m] = np.cos(angle)
            c[10 - m] = np.sin(angle)
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Triplets
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                if time.time() > deadline - 400:
                    break
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = 1.0
                c[10 + m3] = 1.0
                c /= np.linalg.norm(c)
                candidates.append(c)
    
    # Symmetric patterns: only even m
    for m1 in range(0, 11, 2):
        for m2 in range(m1 + 2, 11, 2):
            for m3 in range(m2 + 2, 11, 2):
                if m3 < 11:
                    c = np.zeros(n_m)
                    c[10 + m1] = 1.0
                    c[10 + m2] = 1.0
                    c[10 + m3] = 1.0
                    c /= np.linalg.norm(c)
                    candidates.append(c)
    
    # Random normalized vectors
    for _ in range(200):
        if time.time() > deadline - 400:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(100):
        if time.time() > deadline - 400:
            break
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.6:
                c_rand[10 + m] = rng.standard_normal()
            if m > 0 and rng.random() < 0.3:
                c_rand[10 - m] = rng.standard_normal()
        nrm = np.linalg.norm(c_rand)
        if nrm > 1e-12:
            c_rand /= nrm
            candidates.append(c_rand)
    
    # 7. Evaluate candidates and sort
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 200:
            break
        val = objective_unit(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Apply fixed-point iteration + gradient polish to top candidates
    best_c = None
    best_val = np.inf
    n_to_process = min(60, len(candidate_vals))
    
    for i in range(n_to_process):
        if time.time() > deadline - 30:
            break
        
        val_init, c_init = candidate_vals[i]
        
        # Apply fixed-point iteration with different over-relaxation parameters
        for omega in [0.3, 0.5, 0.7]:
            if time.time() > deadline - 10:
                break
            
            u_fp = fixed_point_iter(c_init, max_iter=300, omega=omega, tol=1e-13)
            val_fp = objective_unit(u_fp)
            
            # Polish with gradient
            u_polished = sphere_polish(u_fp, max_iter=2000, lr_init=0.01, tol=1e-14)
            val_polished = objective_unit(u_polished)
            
            if val_polished < best_val and np.all(np.isfinite(u_polished)):
                best_val = val_polished
                best_c = u_polished.copy()
    
    # 9. Final high-precision polish on the best
    if best_c is not None and time.time() < deadline - 10:
        # Multiple rounds of polishing with decreasing tolerance
        for tol_val in [1e-12, 1e-14, 1e-16]:
            if time.time() > deadline - 5:
                break
            final = sphere_polish(best_c, max_iter=3000, lr_init=0.005, tol=tol_val)
            val_final = objective_unit(final)
            if val_final < best_val:
                best_val = val_final
                best_c = final
    
    # 10. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
