import numpy as np
from scipy.special import lpmv

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
    
    # Precompute weighted Y for faster gradient computation
    # WY = diag(w) @ Y, shape (N_points, dim)
    WY = Y * w[:, None]
    
    def compute_R(c):
        """Compute R = ||f||_4 / ||f||_2"""
        f = Y @ c
        f2 = f * f
        norm2_sq = np.sum(w * f2)
        norm4_sq = np.sum(w * f2 * f2)
        if norm2_sq < 1e-15:
            return np.inf
        return np.sqrt(norm4_sq / norm2_sq)
    
    def compute_grad_quartic(c):
        """
        Compute gradient of J(c) = sum_p w_p (Y_p @ c)^4
        grad_i = 4 * sum_p w_p (Y_p @ c)^3 * Y_{p,i}
        """
        f = Y @ c
        f3 = f * f * f
        # grad = 4 * (Y.T @ (w * f3))
        grad = 4.0 * (WY.T @ f3)
        return grad
    
    # 3. Generate diverse initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # Structured initializations based on symmetry
    # m=0 only
    c0 = np.zeros(dim)
    c0[k] = 1.0
    inits.append(c0)
    
    # Pairs of m and -m (cos and sin)
    for m in range(1, k + 1):
        cm = np.zeros(dim)
        cm[m + k] = 1.0 / np.sqrt(2)
        cm[-m + k] = 1.0 / np.sqrt(2)
        inits.append(cm)
        
        # Also try cos-only
        cc = np.zeros(dim)
        cc[m + k] = 1.0
        inits.append(cc)
    
    # Mixed: m=0 and m=2
    for m2 in [2, 4, 6, 8]:
        for alpha in [0.5, 0.7, 0.85]:
            cmix = np.zeros(dim)
            cmix[k] = alpha
            cmix[m2 + k] = np.sqrt(1 - alpha**2)
            inits.append(cmix)
    
    # Random initializations
    for _ in range(200):
        crand = rng.normal(size=dim)
        crand /= np.linalg.norm(crand)
        inits.append(crand)
    
    # 4. Gradient descent on the sphere with line search
    best_c = inits[0].copy()
    best_R = compute_R(best_c)
    
    n_iters = 500
    step_size = 0.01
    momentum = 0.9
    
    for c_init in inits:
        c = c_init.copy()
        c /= np.linalg.norm(c)
        v = np.zeros(dim)
        
        for it in range(n_iters):
            # Compute gradient of quartic form
            grad = compute_grad_quartic(c)
            
            # Project gradient to tangent space of sphere
            grad_tan = grad - np.dot(grad, c) * c
            
            # Momentum update
            v = momentum * v - step_size * grad_tan
            v_tan = v - np.dot(v, c) * c
            
            # Line search
            alpha = step_size
            for _ in range(20):
                c_new = c + alpha * v_tan
                c_new /= np.linalg.norm(c_new)
                
                R_new = compute_R(c_new)
                if R_new < best_R or R_new < compute_R(c):
                    c = c_new
                    break
                alpha *= 0.5
            
            # Update best
            R_curr = compute_R(c)
            if R_curr < best_R:
                best_R = R_curr
                best_c = c.copy()
            
            # Adaptive step size
            if R_curr < best_R:
                step_size = min(step_size * 1.1, 0.1)
            else:
                step_size = max(step_size * 0.9, 0.001)
            
            # Convergence check
            if np.linalg.norm(v_tan) < 1e-12:
                break
    
    # 5. Final polish with a few more iterations from best
    c = best_c.copy()
    step_size = 0.005
    for _ in range(200):
        grad = compute_grad_quartic(c)
        grad_tan = grad - np.dot(grad, c) * c
        
        c_new = c - step_size * grad_tan
        c_new /= np.linalg.norm(c_new)
        
        R_new = compute_R(c_new)
        if R_new < compute_R(c):
            c = c_new
            if R_new < best_R:
                best_R = R_new
                best_c = c.copy()
    
    # Validation
    if not np.all(np.isfinite(best_c)) or np.max(np.abs(best_c)) < 1e-3 or np.max(np.abs(best_c)) > 1e5:
        best_c = np.zeros(dim)
        best_c[k] = 1.0
    
    return best_c
