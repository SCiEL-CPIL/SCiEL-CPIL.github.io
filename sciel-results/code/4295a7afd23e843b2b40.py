# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_gaunt():
    """
    Precompute Gaunt coefficients for degree-10 spherical harmonics.
    G[a,b,c] = integral Y_a * Y_b * Y_c dΩ (for real orthonormal harmonics)
    
    We need ||f||_4^4 = integral f^4 dΩ where f = sum_m c_m Y_m
    = sum_{a,b,c,d} c_a c_b c_c c_d * G[a,b,c,d]
    
    Using symmetry: G[a,b,c,d] = integral Y_a Y_b Y_c Y_d dΩ
    This can be computed via products of 3-j symbols or by quadrature.
    
    For efficiency, we'll use the quadrature-based approach but with the 
    exact 70x130 grid (which is exact for degree 10 products up to degree 40).
    """
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 2 * k + 1
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Compute P_10^m(x) for m=0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    # Normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    # Build B matrix (n_pts x dim)
    B = np.zeros((n_theta * n_phi, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
    
    # Precompute B^T W B (the L2 Gram matrix, should be identity for orthonormal basis)
    BWB = B.T @ (weights[:, None] * B)  # (dim, dim)
    
    # Precompute B^T W B^2 for L4 norm computation
    # ||f||_4^4 = sum_i w_i f_i^4
    # = c^T (B^T W B^{(3)}) c where B^{(3)}_i = B_i * f_i^2
    # More efficiently: precompute the 4th-order tensor via outer products
    # T[a,b,c,d] = sum_i w_i B[i,a] B[i,b] B[i,c] B[i,d]
    
    # This is a 21^4 tensor = 194481 elements - manageable
    # Build it efficiently using matrix operations
    # T = sum_i w_i (B_i ⊗ B_i ⊗ B_i ⊗ B_i)
    
    # More efficient: for each pair (a,b), precompute sum_i w_i B[i,a]B[i,b]B[i,c]B[i,d]
    # Use the fact that we can write T as a matrix in (ab) vs (cd) form
    
    # Actually, let's precompute the "L4 matrix" M where:
    # ||f||_4^4 = sum_{a,b,c,d} c_a c_b c_c c_d T[a,b,c,d]
    # We can reorganize: for fixed c,d, define v_cd = T[:, :, c, d]
    # Then ||f||_4^4 = sum_{c,d} c_c c_d (c^T v_cd c)
    
    # Let's precompute: for each pair (i,j) in the flattened 21x21, 
    # we need the bilinear form. Actually, let's just precompute:
    # M4 = B^T * diag(W) * (B * f^2) but that depends on f...
    
    # Better approach: precompute the 4th-order tensor as a 2D matrix
    # Reshape T into a (dim*dim) x (dim*dim) matrix
    # T_mat[(a*b), (c*d)] = T[a,b,c,d]
    
    # Compute T efficiently:
    # T[a,b,c,d] = sum_i w_i B[i,a] B[i,b] C[i,c] C[i,d] where C = B*f^2... no
    # T[a,b,c,d] = sum_i w_i B[i,a] B[i,b] B[i,c] B[i,d]
    
    # Use: for each i, B[i,:] is a vector v_i of length dim
    # T = sum_i w_i (v_i ⊗ v_i ⊗ v_i ⊗ v_i)
    # Reshape: T_mat = sum_i w_i (v_i ⊗ v_i) ⊗ (v_i ⊗ v_i)
    # where v_i ⊗ v_i is a dim*dim vector
    
    dim2 = dim * dim
    T_mat = np.zeros((dim2, dim2))
    
    # For each quadrature point, compute outer product of (B_i ⊗ B_i) with itself
    for i in range(n_theta * n_phi):
        vi = B[i, :]  # (dim,)
        vi_outer = np.outer(vi, vi)  # (dim, dim)
        vi_flat = vi_outer.ravel()  # (dim*dim,)
        T_mat += weights[i] * np.outer(vi_flat, vi_flat)
    
    return B, weights, T_mat, BWB

B_GLOBAL, W_GLOBAL, T_MAT, BWB = _precompute_gaunt()
N_PTS = B_GLOBAL.shape[0]
DIM = 21
DIM2 = DIM * DIM

def _compute_ratio_spectral(c):
    """
    Compute R = ||f||_4 / ||f||_2 using spectral (Gaunt coefficient) formula.
    ||f||_2^2 = c^T (B^T W B) c = c^T I c = ||c||^2 (for orthonormal basis)
    ||f||_4^4 = c^T T_mat c (reshaped appropriately)
    """
    # L2 norm squared (using exact orthonormality)
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    
    # L4 norm to the 4th power
    c_flat = c.ravel()  # already flat
    # ||f||_4^4 = sum_{a,b,c,d} c_a c_b c_c c_d T[a,b,c,d]
    # = (c ⊗ c)^T T_mat (c ⊗ c) where c ⊗ c is the outer product flattened
    c_outer = np.outer(c, c)  # (dim, dim)
    c_outer_flat = c_outer.ravel()  # (dim*dim,)
    norm4_4 = c_outer_flat @ T_MAT @ c_outer_flat
    
    if norm4_4 < 0:
        norm4_4 = 0.0
    
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2

def _objective_and_grad_spectral(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its exact gradient using spectral formula.
    
    R = ||f||_4 / ||f||_2
    ||f||_2^2 = c^T c (orthonormal basis)
    ||f||_4^4 = (c⊗c)^T T (c⊗c)
    
    Let A = ||f||_4^4, B = ||f||_2^2 = c^T c
    R = A^{1/4} / B^{1/2}
    
    dR/dc = (1/4) A^{-3/4} (dA/dc) / B^{1/2} - A^{1/4} (1/2) B^{-3/2} (dB/dc)
    
    dA/dc_a = sum_{b,c,d} c_b c_c c_d T[a,b,c,d] + sum_{b,c,d} c_b c_c c_d T[b,a,c,d]
             + sum_{b,c,d} c_b c_c c_d T[b,c,a,d] + sum_{b,c,d} c_b c_c c_d T[b,c,d,a]
    
    Due to symmetry of T: T[a,b,c,d] = T[b,a,c,d] = T[a,b,d,c] = T[b,a,d,c] etc.
    Actually T is symmetric in all indices (it's a 4th order tensor from a product).
    
    More carefully: A = sum_{a,b,c,d} c_a c_b c_c c_d T[a,b,c,d]
    dA/da = sum_{b,c,d} c_b c_c c_d [T[a,b,c,d] + T[b,a,c,d] + T[b,c,a,d] + T[b,c,d,a]]
    
    Since T is symmetric in all pairs: T[a,b,c,d] = T[b,a,c,d] = T[a,b,d,c] = T[b,a,d,c]
    and also T[a,b,c,d] = T[c,d,a,b] (from the construction as sum of outer products)
    
    So dA/da = 4 * sum_{b,c,d} c_b c_c c_d T[a,b,c,d] (by symmetry)
    
    dB/da = 2 * c_a
    
    Let's compute this efficiently.
    """
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    
    c_outer = np.outer(c, c)
    c_outer_flat = c_outer.ravel()
    norm4_4 = c_outer_flat @ T_MAT @ c_outer_flat
    
    if norm4_4 < 0:
        norm4_4 = 0.0
    if norm4_4 < 1e-30:
        return np.inf, np.zeros_like(c)
    
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    # Compute gradient of A = (c⊗c)^T T (c⊗c)
    # dA/da = 4 * sum_{b,c,d} c_b c_c c_d T[a,b,c,d]
    # 
    # Efficient computation:
    # Let u = T_MAT @ c_outer_flat  (this is a vector of length dim*dim)
    # u[(b,c)] = sum_{a,d} T[a,b,c,d] c_a c_d... 
    # Actually, let me think more carefully.
    #
    # A = sum_{a,b,c,d} c_a c_b c_c c_d T[a,b,c,d]
    # 
    # Define: M[a, (b,c)] = sum_d c_d T[a,b,c,d]  -- but this is getting complex
    #
    # Simpler: dA/da = 4 * (T @ (c ⊗ c ⊗ c ⊗ c))_a by symmetry
    # 
    # Let's compute: for each a, dA/da = 4 * sum_{b,c,d} c_b c_c c_d T[a,b,c,d]
    #
    # We can compute this as:
    # temp1[a,b] = sum_c c_c * sum_d c_d * T[a,b,c,d]
    # dA/da = 4 * sum_b c_b * temp1[a,b]
    #
    # Or more efficiently using the matrix form:
    # Let V = c ⊗ c (outer product, shape dim x dim), flat is c_outer_flat
    # T_MAT @ V_flat gives a vector w of length dim*dim
    # w[(a,b)] = sum_{c,d} T[a,b,c,d] * V_flat[(c,d)] = sum_{c,d} T[a,b,c,d] c_c c_d
    # 
    # Then dA/da = 4 * sum_b c_b * w[(a,b)]
    
    w = T_MAT @ c_outer_flat  # (dim*dim,)
    w_reshaped = w.reshape(DIM, DIM)
    
    # dA/da = 4 * sum_b c_b * w_reshaped[a, b]
    dA_dc = 4.0 * (w_reshaped @ c)  # (dim,)
    
    # dB/da = 2 * c_a
    dB_dc = 2.0 * c
    
    # dR/dc = (1/(4*norm4^3 * norm2)) * dA_dc - (norm4 / (2 * norm2^3)) * dB_dc
    grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * dA_dc - (norm4 / (2.0 * norm2**3)) * dB_dc
    
    return R, grad_R

def _normalize_unit_l2(c):
    """Normalize c to unit L2 norm (using exact orthonormality: ||f||_2 = ||c||)."""
    norm = np.linalg.norm(c)
    if norm < 1e-30:
        return c
    return c / norm

def _lgbfsb_optimize(c0, deadline, max_iter=5000):
    """
    Optimize using scipy L-BFGS-B with the spectral objective.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        val, grad = _objective_and_grad_spectral(c)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = scipy_minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': max_iter * 2}
        )
        if result.fun < 1e10:
            c_opt = _normalize_unit_l2(result.x)
            val_opt = _compute_ratio_spectral(c_opt)
            return val_opt, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _gradient_descent_polish(c, deadline, max_iter=2000, lr_init=0.01):
    """
    Refined gradient descent with adaptive step size and backtracking for polishing.
    """
    c = c.copy()
    current_val, grad = _objective_and_grad_spectral(c)
    if not np.isfinite(current_val):
        return current_val, c
    
    best_val = current_val
    best_c = c.copy()
    lr = lr_init
    momentum = np.zeros_like(c)
    mu = 0.9
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Momentum
        momentum = mu * momentum - lr * grad
        d = momentum
        
        # Backtracking line search
        c1 = 1e-4
        rho = 0.5
        step = 1.0
        fx0 = current_val
        
        improved = False
        for _backtrack in range(50):
            c_try = c + step * d
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio_spectral(c_try)
            if np.isfinite(val_try) and val_try <= fx0 + c1 * step * np.dot(grad, d):
                c = c_try
                current_val = val_try
                grad, _ = _objective_and_grad_spectral(c)
                # Recompute grad properly
                _, grad = _objective_and_grad_spectral(c)
                improved = True
                break
            step *= rho
        
        if not improved:
            # Try a small step
            step = 0.001
            c_try = c + step * d
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio_spectral(c_try)
            if np.isfinite(val_try) and val_try < current_val:
                c = c_try
                current_val = val_try
                _, grad = _objective_and_grad_spectral(c)
            else:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_c = c.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    return best_val, best_c

def _local_restart_optimize(c0, deadline, n_restarts=5, noise_scale_init=0.1):
    """
    Local restarts with decaying noise around the best solution.
    """
    best_val, best_c = _lgbfsb_optimize(c0, deadline)
    
    noise_scale = noise_scale_init
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Add decaying noise
        noise = np.random.randn(DIM) * noise_scale
        c_noisy = best_c + noise
        c_noisy = _normalize_unit_l2(c_noisy)
        
        val, c_opt = _lgbfsb_optimize(c_noisy, deadline)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_c = c_opt.copy()
        
        noise_scale *= 0.5
    
    return best_val, best_c

def _generate_initial_points():
    """Generate diverse initial points for optimization."""
    inits = []
    
    # 1. Zonal harmonic (m=0)
    c_zonal = np.zeros(DIM)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    # 2. Pure m=10 (cos)
    c_m10 = np.zeros(DIM)
    c_m10[20] = 1.0
    inits.append(c_m10)
    
    # 3. Pure m=-10 (sin)
    c_mneg10 = np.zeros(DIM)
    c_mneg10[0] = 1.0
    inits.append(c_mneg10)
    
    # 4. m=10 and m=-10 equal
    c_m10_both = np.zeros(DIM)
    c_m10_both[20] = 1.0
    c_m10_both[0] = 1.0
    inits.append(c_m10_both)
    
    # 5. m=9 cos
    c_m9 = np.zeros(DIM)
    c_m9[19] = 1.0
    inits.append(c_m9)
    
    # 6. m=8 cos
    c_m8 = np.zeros(DIM)
    c_m8[18] = 1.0
    inits.append(c_m8)
    
    # 7. Random points
    np.random.seed(42)
    for seed in range(15):
        np.random.seed(seed)
        c = np.random.randn(DIM)
        inits.append(c)
    
    # 8. Sparse combinations
    for i in range(DIM):
        c = np.zeros(DIM)
        c[i] = 1.0
        inits.append(c)
    
    # 9. Pairs
    for i in range(0, DIM, 2):
        c = np.zeros(DIM)
        c[i] = 1.0
        c[i+1] = 1.0
        inits.append(c)
    
    # 10. Alternating
    c_alt = np.zeros(DIM)
    for i in range(DIM):
        c_alt[i] = 1.0 if i % 2 == 0 else -1.0
    inits.append(c_alt)
    
    # Normalize all
    inits_normalized = [_normalize_unit_l2(c) for c in inits]
    
    # Evaluate and sort
    scored = []
    for c in inits_normalized:
        val = _compute_ratio_spectral(c)
        if np.isfinite(val):
            scored.append((val, c))
    
    scored.sort(key=lambda x: x[0])
    return scored

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Generate and evaluate initial points
    scored_inits = _generate_initial_points()
    
    best_val = np.inf
    best_coeffs = None
    
    # Evaluate top starts
    for val, c in scored_inits[:5]:
        if val < best_val:
            best_val = val
            best_coeffs = c.copy()
    
    # Optimize from multiple diverse starts
    n_starts = 12
    for i, (val, c_start) in enumerate(scored_inits[:n_starts]):
        if time.time() > deadline - 30:
            break
        
        try:
            # L-BFGS-B optimization
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, max_iter=3000)
            
            if np.isfinite(val_opt) and val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
        
        # Local restarts with decaying noise
        if time.time() > deadline - 20:
            break
        try:
            val_restart, c_restart = _local_restart_optimize(
                best_coeffs if best_coeffs is not None else c_start, 
                deadline, 
                n_restarts=3,
                noise_scale_init=0.05
            )
            if np.isfinite(val_restart) and val_restart < best_val:
                best_val = val_restart
                best_coeffs = c_restart.copy()
        except Exception:
            pass
        
        # Gradient descent polish
        if time.time() > deadline - 15:
            break
        try:
            val_polish, c_polish = _gradient_descent_polish(
                best_coeffs if best_coeffs is not None else c_start,
                deadline,
                max_iter=1000,
                lr_init=0.005
            )
            if np.isfinite(val_polish) and val_polish < best_val:
                best_val = val_polish
                best_coeffs = c_polish.copy()
        except Exception:
            pass
    
    # Final ultra-fine polish
    if best_coeffs is not None and time.time() < deadline - 10:
        try:
            val_final, c_final = _gradient_descent_polish(
                best_coeffs,
                deadline,
                max_iter=3000,
                lr_init=0.001
            )
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best_coeffs = c_final.copy()
        except Exception:
            pass
    
    # Final validation
    if best_coeffs is None:
        best_coeffs = np.ones(DIM) / np.sqrt(DIM)
    
    coeffs = best_coeffs.astype(np.float64)
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(DIM) / np.sqrt(DIM)
        
    max_c = np.max(np.abs(coeffs))
    if max_c < 1e-3:
        if max_c > 0:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = np.ones(DIM) / np.sqrt(DIM)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
        
    return coeffs

# EVOLVE-BLOCK-END
