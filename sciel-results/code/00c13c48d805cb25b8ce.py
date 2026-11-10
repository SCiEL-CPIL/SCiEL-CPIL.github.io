import numpy as np
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on quadrature grid.
    2. Define objective R(c) and its Euclidean gradient.
    3. Use Riemannian Gradient Descent on the unit sphere ||c||=1.
       This projects the Euclidean gradient onto the tangent space T_c S^20,
       which is the correct geometric approach for scale-invariant optimization.
    4. Multi-start with diverse initializations to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    k = 10
    n_m = 2 * k + 1 # 21 coefficients
    
    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)
    
    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # Precompute weights for norms
    # We want to minimize R = ||f||_4 / ||f||_2
    # Equivalent to minimizing ||f||_4 subject to ||f||_2 = 1
    
    def compute_norms(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        return norm2_sq, norm4_4, f_vals

    def objective_ratio(c):
        norm2_sq, norm4_4, _ = compute_norms(c)
        if norm2_sq < 1e-12:
            return 1e10
        return (norm4_4 ** 0.25) / (norm2_sq ** 0.5)

    def objective_and_grad(c):
        norm2_sq, norm4_4, f_vals = compute_norms(c)
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients
        w_f = w_flat * f_vals
        w_f3 = w_flat * f_vals ** 3
        
        sum_wf_A = A.T @ w_f * d_phi
        sum_wf3_A = A.T @ w_f3 * d_phi
        
        grad = (N4**3 / N2) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        return R, grad

    # Multi-start strategy
    best_c = None
    best_val = np.inf
    
    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Random starts
    for _ in range(50):
        c_rand = rng.standard_normal(n_m)
        candidates.append(c_rand / np.linalg.norm(c_rand))
        
    # Single mode starts
    for m in range(-10, 11):
        c_single = np.zeros(n_m)
        c_single[m + 10] = 1.0
        candidates.append(c_single)
        
    # Pair starts (common in flat harmonics)
    for m1 in range(-10, 11, 2):
        for m2 in range(m1 + 1, 11, 2):
            c_pair = np.zeros(n_m)
            c_pair[m1 + 10] = 1.0
            c_pair[m2 + 10] = 1.0
            candidates.append(c_pair / np.linalg.norm(c_pair))
            
    # Evaluate candidates to find top starts
    top_candidates = []
    for c in candidates:
        val = objective_ratio(c)
        top_candidates.append((val, c))
    top_candidates.sort(key=lambda x: x[0])
    top_candidates = top_candidates[:20] # Use top 20 for Riemannian descent

    # Riemannian Gradient Descent
    # We optimize on the sphere S^{n_m-1}
    # Retraction: R_c(v) = c + v, then normalize.
    # This is simple and effective for this problem.
    
    for val, c_init in top_candidates:
        if time.time() > deadline - 20:
            break
            
        c = c_init.copy()
        current_val = objective_ratio(c)
        
        # Adaptive step size
        step_size = 0.1
        min_step = 1e-6
        
        # Simple backtracking line search
        for it in range(500):
            if time.time() > deadline - 5:
                break
                
            val, grad = objective_and_grad(c)
            
            # Check for convergence
            if abs(val - current_val) < 1e-10 and it > 10:
                break
            current_val = val
            
            # Project gradient to tangent space
            # T_c S = {v | <v, c> = 0}
            # P_c(g) = g - <g, c> c
            g_dot_c = np.dot(grad, c)
            grad_tan = grad - g_dot_c * c
            
            # Check gradient norm
            grad_norm = np.linalg.norm(grad_tan)
            if grad_norm < 1e-8:
                break
            
            # Normalize gradient for step size control
            grad_tan_norm = grad_tan / grad_norm
            
            # Backtracking line search
            alpha = step_size
            improved = False
            for _ in range(20):
                # Retraction: move in direction, then project back to sphere
                c_new = c + alpha * grad_tan_norm
                norm_c_new = np.linalg.norm(c_new)
                if norm_c_new < 1e-12:
                    alpha *= 0.5
                    continue
                c_new = c_new / norm_c_new
                
                val_new = objective_ratio(c_new)
                
                if val_new < val:
                    improved = True
                    break
                alpha *= 0.5
                
            if not improved:
                # If we can't improve with this step, try a smaller step
                # or break if step is too small
                if alpha < min_step:
                    break
                # Try a random perturbation if stuck?
                # For now, just break and move to next start
                break
            
            c = c_new
            # Gradually decrease step size if we've been improving steadily
            # Or keep it constant. Let's keep it constant but allow it to grow slightly
            # if we're making large improvements? No, standard is constant or decreasing.
            # Let's try a simple decay: step_size *= 0.99
            step_size *= 0.995
            
        val_final = objective_ratio(c)
        if val_final < best_val:
            best_val = val_final
            best_c = c.copy()

    # Final Polish with BFGS on the ratio directly (with gradient)
    # This will refine the point found by Riemannian descent
    
    if best_c is not None and time.time() < deadline - 10:
        try:
            # We need to handle the constraint ||c||=1.
            # BFGS doesn't handle constraints natively.
            # We can use a penalty method or just rely on the Riemannian result.
            # Given the Riemannian descent is quite accurate, let's try a few
            # Newton steps on the sphere manually or just return the Riemannian result.
            # Actually, let's try L-BFGS-B with a penalty term for ||c||^2 - 1
            # or just use the Riemannian result which is already on the sphere.
            
            # Let's do a few more Riemannian steps with a smaller step size
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                val, grad = objective_and_grad(best_c)
                g_dot_c = np.dot(grad, best_c)
                grad_tan = grad - g_dot_c * best_c
                grad_norm = np.linalg.norm(grad_tan)
                if grad_norm < 1e-10:
                    break
                grad_tan_norm = grad_tan / grad_norm
                
                # Small step
                c_new = best_c + 0.01 * grad_tan_norm
                norm_c_new = np.linalg.norm(c_new)
                if norm_c_new < 1e-12:
                    break
                c_new = c_new / norm_c_new
                val_new = objective_ratio(c_new)
                if val_new < val:
                    best_c = c_new
                    best_val = val_new
                else:
                    break
        except Exception:
            pass

    # Final check on validity
    if best_c is None or not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
