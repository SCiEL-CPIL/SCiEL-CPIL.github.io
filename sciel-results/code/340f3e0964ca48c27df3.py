import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses sphere-constrained optimization with Newton refinement and diverse
    initial candidates exploiting m-mode structure.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

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
    
    # m = 0
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, 10] = N * P
    
    # m > 0 and m < 0
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, m + 10] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, -m + 10] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis and quadratic forms
    B = A * w_flat[:, None]  # B[i,j] = w_i * A[i,j]
    # Precompute A^T B for fast gradient: (A^T B)[j,l] = sum_i A[i,j] * w_i * A[i,l]
    ATB = A.T @ B  # n_m x n_m
    
    # 3. Objective and gradient
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradient of R with respect to u (on the sphere)
        # dR/du = (N4^3) * (1/(4*N4^4)) * 4*f^3 * A - (N4/N2) * (1/(2*N2)) * 2*f * A
        #       = (N4^2) * f^3 * A / N4^4 - (N4/(2*N2^2)) * f * A
        #       = f^3 * A / N4^2 - N4 * f * A / (2 * N2^2)
        # But we need to be more careful. Let's use the chain rule properly.
        
        # R = N4 / N2 where N4 = (sum w f^4)^{1/4}, N2 = (sum w f^2)^{1/2}
        # dR/d(f_i) = (1/N2) * dN4/df_i - (N4/N2^2) * dN2/df_i
        # dN4/df_i = (1/4) * N4^{-3} * 4 * w_i * f_i^3 = w_i * f_i^3 / N4^3
        # dN2/df_i = (1/2) * N2^{-1} * 2 * w_i * f_i = w_i * f_i / N2
        # So dR/df_i = w_i * f_i^3 / (N2 * N4^3) - N4 * w_i * f_i / (N2^3)
        #            = w_i * (f_i^3 / N4^3 - N4 * f_i / N2^3) / N2
        
        # Then dR/du_j = sum_i dR/df_i * A[i,j]
        
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        sum_wf_A = d_phi * (B.T @ f_vals)
        
        grad_u = (1.0 / N2) * (sum_wf3_A / N4**3 - (N4 / N2**3) * sum_wf_A)
        
        # Project onto tangent space of sphere
        u_dot_grad = np.dot(u, grad_u)
        grad_tangent = grad_u - u * u_dot_grad
        
        # Chain rule: dR/dc = grad_tangent / norm_c (since u = c/norm_c)
        grad_c = grad_tangent / norm_c
        
        return R, grad_c

    # 4. Sphere-constrained gradient descent with Newton-like refinement
    def sphere_polish(c, max_iter=1000, lr_init=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = d_phi * np.sum(w_flat * f2)
            norm4_4 = d_phi * np.sum(w_flat * f4)
            if norm2_sq < 1e-12:
                break
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            R_curr = N4 / N2
            
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            sum_wf_A = d_phi * (B.T @ f_vals)
            
            grad_u = (1.0 / N2) * (sum_wf3_A / N4**3 - (N4 / N2**3) * sum_wf_A)
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            
            # Backtracking line search with Armijo condition
            step = lr
            R_curr_val = R_curr
            improved = False
            for _ in range(30):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                f_new = A @ u_new
                f2_new = f_new ** 2
                f4_new = f2_new ** 2
                norm2_sq_new = d_phi * np.sum(w_flat * f2_new)
                norm4_4_new = d_phi * np.sum(w_flat * f4_new)
                if norm2_sq_new < 1e-12:
                    step *= 0.5
                    continue
                R_new = (norm4_4_new ** 0.25) / np.sqrt(norm2_sq_new)
                
                # Armijo condition
                if R_new < R_curr_val - 1e-4 * step * grad_norm:
                    u = u_new
                    R_curr_val = R_new
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved or step < 1e-14:
                break
            # Adaptive learning rate
            lr = max(lr * 0.999, 0.0005)
        
        return u

    # 5. High-precision Newton-like polish: use local quadratic approximation
    def newton_polish(c, max_iter=500):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = d_phi * np.sum(w_flat * f2)
            norm4_4 = d_phi * np.sum(w_flat * f4)
            if norm2_sq < 1e-12:
                break
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            R_curr = N4 / N2
            
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            sum_wf_A = d_phi * (B.T @ f_vals)
            
            grad_u = (1.0 / N2) * (sum_wf3_A / N4**3 - (N4 / N2**3) * sum_wf_A)
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-15:
                break
            
            # Approximate Hessian via finite differences in tangent directions
            # Use a simple diagonal approximation: scale gradient by local curvature estimate
            # For the L4/L2 ratio, the local curvature can be estimated from the second derivative
            # along the gradient direction
            
            # Try a Newton-like step: use the ratio of gradient to approximate Hessian
            # H_approx ~ grad_norm / max_step (heuristic)
            max_step = 0.1
            hessian_scale = grad_norm / max_step if grad_norm > 1e-12 else 1.0
            
            # Damped Newton step
            step_size = np.dot(grad_tangent, grad_tangent) / (hessian_scale * max_step)
            step_size = min(step_size, 0.05)
            
            u_new = u - step_size * grad_tangent
            u_new_norm = np.linalg.norm(u_new)
            if u_new_norm < 1e-12:
                break
            u_new = u_new / u_new_norm
            
            f_new = A @ u_new
            f2_new = f_new ** 2
            f4_new = f2_new ** 2
            norm2_sq_new = d_phi * np.sum(w_flat * f2_new)
            norm4_4_new = d_phi * np.sum(w_flat * f4_new)
            if norm2_sq_new < 1e-12:
                break
            R_new = (norm4_4_new ** 0.25) / np.sqrt(norm2_sq_new)
            
            if R_new < R_curr:
                u = u_new
            else:
                # Fallback to smaller step
                for factor in [0.5, 0.25, 0.1]:
                    u_try = u - factor * step_size * grad_tangent
                    u_try_norm = np.linalg.norm(u_try)
                    if u_try_norm < 1e-12:
                        continue
                    u_try = u_try / u_try_norm
                    f_try = A @ u_try
                    f2_try = f_try ** 2
                    f4_try = f2_try ** 2
                    norm2_sq_try = d_phi * np.sum(w_flat * f2_try)
                    norm4_4_try = d_phi * np.sum(w_flat * f4_try)
                    if norm2_sq_try < 1e-12:
                        continue
                    R_try = (norm4_4_try ** 0.25) / np.sqrt(norm2_sq_try)
                    if R_try < R_curr:
                        u = u_try
                        break
        
        return u

    # 6. Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Pure m>0 cos modes
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
    
    # Pure m<0 sin modes
    for m in range(1, 11):
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
    
    # Pairs of low-m modes (focus on m <= 6)
    for m1 in range(0, 7):
        for m2 in range(m1 + 1, 8):
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
    
    # Pairs including sin modes
    for m1 in range(0, 7):
        for m2 in range(1, 7):
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[-m2 + 10] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
    
    # Triplets of low-m modes
    for m1 in range(0, 5):
        for m2 in range(m1 + 1, 7):
            for m3 in range(m2 + 1, 8):
                c_triple = np.zeros(n_m)
                c_triple[10 + m1] = 1.0
                c_triple[10 + m2] = 1.0
                c_triple[10 + m3] = 1.0
                c_triple /= np.linalg.norm(c_triple)
                candidates.append(c_triple)
    
    # Quadruplets of very low-m modes
    for m1 in range(0, 4):
        for m2 in range(m1 + 1, 6):
            for m3 in range(m2 + 1, 7):
                for m4 in range(m3 + 1, 8):
                    c_quad = np.zeros(n_m)
                    c_quad[10 + m1] = 1.0
                    c_quad[10 + m2] = 1.0
                    c_quad[10 + m3] = 1.0
                    c_quad[10 + m4] = 1.0
                    c_quad /= np.linalg.norm(c_quad)
                    candidates.append(c_quad)
    
    # Random normalized vectors with low-m emphasis
    for _ in range(120):
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            weight = 1.0 / (1.0 + 0.3 * m)
            if rng.random() < 0.85:
                c_rand[10 + m] = rng.standard_normal() * weight
            if m > 0 and rng.random() < 0.5:
                c_rand[-m + 10] = rng.standard_normal() * weight
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)
    
    # Structured: exponential decay in m with random phases
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for _ in range(3):
            c_struct = np.zeros(n_m)
            for m in range(0, 11):
                c_struct[10 + m] = alpha ** m * rng.standard_normal()
                if m > 0:
                    c_struct[-m + 10] = alpha ** m * rng.standard_normal()
            if np.linalg.norm(c_struct) > 1e-12:
                c_struct /= np.linalg.norm(c_struct)
                candidates.append(c_struct)
    
    # Gaussian-like concentration patterns
    for center in [0, 2, 4, 6, 8, 10]:
        for width in [1.0, 1.5, 2.0, 3.0]:
            c_gauss = np.zeros(n_m)
            for m in range(0, 11):
                c_gauss[10 + m] = np.exp(-0.5 * ((m - center) / width) ** 2) * rng.standard_normal()
                if m > 0:
                    c_gauss[-m + 10] = np.exp(-0.5 * ((m - center) / width) ** 2) * rng.standard_normal()
            if np.linalg.norm(c_gauss) > 1e-12:
                c_gauss /= np.linalg.norm(c_gauss)
                candidates.append(c_gauss)
    
    # Symmetric patterns: equal weights on symmetric m pairs
    for _ in range(30):
        c_sym = np.zeros(n_m)
        for m in range(0, 11):
            amp = rng.standard_normal() / (1.0 + 0.2 * m)
            c_sym[10 + m] = amp
            if m > 0:
                c_sym[-m + 10] = amp * rng.choice([-1, 1])
        if np.linalg.norm(c_sym) > 1e-12:
            c_sym /= np.linalg.norm(c_sym)
            candidates.append(c_sym)

    # 7. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 120:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Optimize top candidates with adaptive budget
    best_c = None
    best_val = np.inf
    n_to_optimize = min(25, len(candidate_vals))
    
    for i in range(n_to_optimize):
        if time.time() > deadline - 90:
            break
        val_init, c_init = candidate_vals[i]
        
        # Skip if initial value is already much worse than current best
        if best_val < np.inf and val_init > best_val * 1.2:
            continue
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # Quick sphere polish on the best result so far
        if best_c is not None and time.time() < deadline - 45:
            polished = sphere_polish(best_c, max_iter=600, lr_init=0.06)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 9. Final high-precision polish on the best candidate
    if best_c is not None and time.time() < deadline - 8:
        # Multiple polish passes with decreasing learning rate
        for lr in [0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
            if time.time() > deadline - 8:
                break
            polished = sphere_polish(best_c, max_iter=1000, lr_init=lr)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()
        
        # Newton-like polish for final refinement
        if time.time() < deadline - 4:
            polished = newton_polish(best_c, max_iter=300)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()
        
        # One more ultra-fine polish if time permits
        if time.time() < deadline - 2:
            polished = sphere_polish(best_c, max_iter=2000, lr_init=0.0005)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()
        
        # Final micro-polish
        if time.time() < deadline - 1:
            polished = sphere_polish(best_c, max_iter=1000, lr_init=0.0002)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 10. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
