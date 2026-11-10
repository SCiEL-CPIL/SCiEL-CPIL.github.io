import numpy as np
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Construct the weighted basis matrix M such that M^T M = I.
    2. Minimize sum((M c)^4) subject to ||c||_2 = 1.
    3. Use fixed-point iteration c <- normalize(M^T (M c)^3) followed by projected gradient descent.
    4. Multiple restarts to avoid local minima.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

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
    
    # Effective weights for the discrete inner product
    weights = w_flat * d_phi

    # 2. Construct Basis Functions A
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
    
    # Construct M
    sqrt_w = np.sqrt(weights)
    M = A * sqrt_w[:, None]
    
    # Precompute M.T for efficiency
    MT = M.T

    def objective(c):
        y = M @ c
        return np.sum(y**4)

    def grad(c):
        y = M @ c
        return 4.0 * (MT @ (y**3))

    def fixed_point_step(c):
        """Euler-Lagrange fixed point: c_new ∝ M^T (M c)^3"""
        y = M @ c
        y3 = y**3
        c_new = MT @ y3
        norm = np.linalg.norm(c_new)
        if norm < 1e-12:
            return c
        return c_new / norm

    def polish(c, max_iter=2000, lr=0.01):
        """Projected gradient descent with backtracking line search"""
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        c = c / norm_c
        
        f_curr = objective(c)
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            g = grad(c)
            # Project gradient to tangent space of unit sphere
            c_dot_g = np.dot(c, g)
            grad_t = g - c_dot_g * c
            
            # Line search with backtracking
            alpha = lr
            success = False
            for _ in range(25):
                c_new = c - alpha * grad_t
                c_new_norm = np.linalg.norm(c_new)
                if c_new_norm < 1e-12:
                    alpha *= 0.5
                    continue
                c_new = c_new / c_new_norm
                f_new = objective(c_new)
                
                if f_new < f_curr:
                    c = c_new
                    f_curr = f_new
                    success = True
                    break
                else:
                    alpha *= 0.5
            
            if not success:
                break
                
            # Check convergence
            if np.linalg.norm(grad_t) < 1e-12:
                break
        
        return c

    # 3. Search with diverse initial candidates
    best_c = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Strategy 1: Pure modes (single harmonic)
    for m in range(n_m):
        c = np.zeros(n_m)
        c[m] = 1.0
        candidates.append(c)
    
    # Strategy 2: Pairs of modes (exploit symmetry)
    for i in range(n_m):
        for j in range(i+1, n_m):
            c = np.zeros(n_m)
            c[i] = 1.0
            c[j] = 1.0
            c /= np.linalg.norm(c)
            candidates.append(c)
    
    # Strategy 3: Symmetric combinations (cos/sin pairs)
    for m in range(1, k+1):
        idx_pos = m + 10
        idx_neg = -m + 10
        c = np.zeros(n_m)
        c[idx_pos] = 1.0
        c[idx_neg] = 1.0
        c /= np.linalg.norm(c)
        candidates.append(c)
        
        c = np.zeros(n_m)
        c[idx_pos] = 1.0
        c[idx_neg] = -1.0
        c /= np.linalg.norm(c)
        candidates.append(c)
    
    # Strategy 4: Random with structure (emphasize certain m values)
    for _ in range(500):
        if time.time() > deadline - 150:
            break
        c = rng.standard_normal(n_m)
        # Bias toward lower |m| modes which may have better L4/L2
        for m_idx in range(n_m):
            m_val = abs(m_idx - 10)
            if m_val > 5:
                c[m_idx] *= 0.5
        c /= np.linalg.norm(c)
        candidates.append(c)
    
    # Strategy 5: Pure random
    for _ in range(500):
        if time.time() > deadline - 150:
            break
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        candidates.append(c)
    
    # Sort by initial objective value
    cand_objs = []
    for c in candidates:
        val = objective(c)
        cand_objs.append((val, c))
    
    cand_objs.sort(key=lambda x: x[0])
    
    # Process top candidates with fixed-point + polish
    n_to_process = min(150, len(cand_objs))
    
    for i in range(n_to_process):
        if time.time() > deadline - 15:
            break
            
        val_init, c_init = cand_objs[i]
        
        # Fixed point iterations (more aggressive for promising starts)
        c_fp = c_init.copy()
        n_fp_iters = 80 if i < 20 else 40
        for _ in range(n_fp_iters):
            if time.time() > deadline - 5:
                break
            c_fp = fixed_point_step(c_fp)
            
        # Polish
        c_pol = polish(c_fp, max_iter=3000)
        val_pol = objective(c_pol)
        
        if val_pol < best_val:
            best_val = val_pol
            best_c = c_pol.copy()
            
    # Continue with more random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 30:
        c = rng.standard_normal(n_m)
        c /= np.linalg.norm(c)
        
        # Fixed point
        c_fp = c.copy()
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            c_fp = fixed_point_step(c_fp)
            
        # Polish
        c_pol = polish(c_fp, max_iter=1500)
        val_pol = objective(c_pol)
        
        if val_pol < best_val:
            best_val = val_pol
            best_c = c_pol.copy()
        
        restart_count += 1

    if best_c is None:
        best_c = np.zeros(n_m)
        best_c[10] = 1.0

    # Scale to valid range
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
