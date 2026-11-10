# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    B = np.zeros((n_pts, dim))
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
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
        
    return B, weights

B_GLOBAL, W_GLOBAL = _precompute_basis()

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _objective_and_grad(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient for coefficients c.
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    
    R = norm4 / norm2
    
    # Gradient of ||f||_4
    # d/dc (sum w f^4)^0.25 = 0.25 (sum w f^4)^-0.75 * 4 sum w f^3 B
    # = (sum w f^4)^-0.75 sum w f^3 B
    # Let S4 = sum w f^4. d/dc ||f||_4 = S4^-0.75 * (B.T @ (w * f^3))
    
    # Gradient of ||f||_2
    # d/dc (sum w f^2)^0.5 = 0.5 (sum w f^2)^-0.5 * 2 sum w f B
    # = S2^-0.5 * (B.T @ (w * f))
    
    grad_4_term = (B_GLOBAL.T @ (W_GLOBAL * f * f2))
    grad_2_term = (B_GLOBAL.T @ (W_GLOBAL * f))
    
    # R = N4 / N2
    # dR = (1/N2) dN4 - (N4/N2^2) dN2
    # dN4 = N4^3 * S4^-0.75 * grad_4_term  ... wait.
    # N4 = S4^0.25
    # dN4/dc = 0.25 S4^-0.75 * (4 B.T @ (w f^3)) = S4^-0.75 * B.T @ (w f^3)
    # So dN4 = N4^3 * S4^-0.75 * grad_4_term? No.
    # dN4 = (dN4/dc) . dc.
    # dN4/dc = S4^(-3/4) * (B.T @ (w f^3))
    
    # dN2/dc = S2^(-1/2) * (B.T @ (w f))
    
    grad_R = (1.0 / norm2) * (norm4**3 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))) - (norm4 / (norm2**3)) * (2.0 * (B_GLOBAL.T @ (W_GLOBAL * f)))
    
    # Correction: 
    # N4 = (sum w f^4)^0.25
    # dN4/dx = 0.25 (sum w f^4)^-0.75 * 4 sum w f^3 (df/dx) = (sum w f^4)^-0.75 sum w f^3 (df/dx)
    # Let G4 = B.T @ (w f^3). Then dN4 = N4^3 * (N4**3 / N4^3? No)
    # N4^3 = (sum w f^4)^0.75.
    # So dN4/dc = (sum w f^4)^-0.75 * G4 = N4^-3 * G4? No.
    # (sum w f^4)^-0.75 = (N4^4)^-0.75 = N4^-3.
    # So dN4/dc = N4^-3 * G4? No, N4 = S4^0.25. S4 = N4^4.
    # S4^-0.75 = (N4^4)^-0.75 = N4^-3.
    # So dN4/dc = N4^-3 * G4.
    # Wait, G4 = B.T @ (w f^3).
    
    # Let's re-verify.
    # N4 = S4^{1/4}. dN4 = (1/4) S4^{-3/4} dS4.
    # dS4 = sum 4 w f^3 df = 4 sum w f^3 (B dc).
    # dS4/dc = 4 B^T (w f^3) = 4 G4.
    # dN4/dc = (1/4) S4^{-3/4} * 4 G4 = S4^{-3/4} G4.
    # S4^{-3/4} = (N4^4)^{-3/4} = N4^{-3}.
    # So dN4/dc = N4^{-3} G4? No, N4 = S4^{1/4} => S4 = N4^4.
    # S4^{-3/4} = (N4^4)^{-3/4} = N4^{-3}.
    # So dN4/dc = N4^{-3} * G4? 
    # Let's check dimensions. N4 is scalar. G4 is vector.
    # If f is constant 1 on unit sphere (area 4pi), S4 = 4pi. N4 = (4pi)^0.25.
    # G4 = B^T (w * 1) = 0 if B is orthonormal and w is quadrature? 
    # For m=0, Y0 is constant. B^T (w Y0) = ||Y0||^2 = 1.
    # So for c corresponding to Y0, G4 = 1.
    # dN4/dc = N4^-3 * 1.
    # N4 = (4pi * N0^4 * 4pi)^0.25? No.
    # If c = e_0, f = Y0.
    # ||f||_4 = (sum w Y0^4)^0.25 = (Y0^4 * 4pi)^0.25 = Y0 * (4pi)^0.25.
    # Y0 = N0.
    # So N4 = N0 (4pi)^0.25.
    # dN4/dc = N4^-3 * G4.
    # G4 = e_0^T B^T (w Y0^3) = (B e_0)^T (w Y0^3) = Y0^T (w Y0^3) = sum w Y0^4 = S4 = N4^4.
    # So dN4/dc = N4^-3 * N4^4 = N4.
    # Correct.
    
    # Similarly for N2.
    # N2 = S2^0.5. dN2/dc = 0.5 S2^-0.5 * 2 B^T (w f) = S2^-0.5 G2.
    # S2^-0.5 = N2^-1.
    # dN2/dc = N2^-1 G2.
    
    # R = N4 / N2.
    # dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
    # dR/dc = (1/N2) (N4^-3 G4) - (N4/N2^2) (N2^-1 G2)
    # dR/dc = (G4 / (N2 N4^3)) - (N4 G2 / (N2^3))
    
    G4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    G2 = B_GLOBAL.T @ (W_GLOBAL * f)
    
    grad_R = G4 / (norm2 * norm4**3) - (norm4 * G2) / (norm2**3)
    
    return R, grad_R

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _gradient_descent_polish(c, deadline, max_iter=1000):
    """
    Refined gradient descent with adaptive step size and backtracking.
    """
    c = c.copy()
    current_val = _compute_ratio(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        step = 0.1
        improved = False
        for _backtrack in range(50):
            c_try = c - step * grad
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-18:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            if step < 1e-15:
                break
            # Try a larger step if backtracking failed too early
            step = 0.5
            for _backtrack in range(30):
                c_try = c + step * grad
                c_try = _normalize_unit_l2(c_try)
                val_try = _compute_ratio(c_try)
                if val_try < current_val - 1e-18:
                    c = c_try
                    current_val = val_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    return current_val, c

def _high_precision_polish(c, deadline):
    """
    Final high-precision polish: tight-tolerance L-BFGS-B followed by many
    small-step gradient descent iterations to squeeze out remaining error.
    """
    c = c.copy()
    
    # Tight L-BFGS-B
    if time.time() < deadline - 10:
        def obj_and_grad(c):
            if not np.all(np.isfinite(c)):
                return 1e10, np.zeros_like(c)
            c_norm = _normalize_unit_l2(c)
            val, grad = _objective_and_grad(c_norm)
            if not np.isfinite(val):
                return 1e10, np.zeros_like(c)
            return val, grad
        
        try:
            result = scipy_minimize(
                obj_and_grad,
                c,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxfun': 10000,
                         'eps': 1e-12}
            )
            if result.fun < 1e10:
                c = _normalize_unit_l2(result.x)
        except Exception:
            pass
    
    # Many small-step gradient descent
    current_val = _compute_ratio(c)
    for it in range(5000):
        if time.time() > deadline - 3:
            break
        
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-18:
            break
        
        step = 0.05
        improved = False
        for _backtrack in range(60):
            c_try = c - step * grad
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-20:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            if step < 1e-16:
                break

    return current_val, c

def _lgbfsb_optimize(c0, deadline, norm_func=None):
    """
    Optimize using L-BFGS-B with gradient information.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = norm_func(c)
        val, grad = _objective_and_grad(c_norm)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = scipy_minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 5000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _generate_symmetric_initial_points(dim):
    """
    Generate initial points based on direct structural analysis of the problem.
    The goal is to find 'flat' functions.
    """
    inits = []
    
    # 1. Zonal (m=0) - Known to have high peaks, but a baseline.
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    # 2. Pure m=10 (High oscillation in phi, low in theta)
    c_m10 = np.zeros(dim)
    c_m10[20] = 1.0 # m=10
    c_m10[0] = 1.0  # m=-10
    inits.append(c_m10)
    
    # 3. All m equal (Isotropic in m-space)
    c_all = np.ones(dim)
    inits.append(c_all)
    
    # 4. Inverse Normalization weights (Equal L2 contribution per mode)
    # N_m = sqrt((2l+1)/4pi * (l-m)!/(l+m)!)
    # We want c_m * N_m to be constant? No, we want the terms to add constructively destructively?
    # If we set c_m = 1/N_m, then c_m N_m = 1.
    # This makes the "amplitude" of each term equal in the orthonormal basis sense.
    k = 10
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    c_inv_N = np.zeros(dim)
    c_inv_N[10] = 1.0 / N[0]
    for m in range(1, 11):
        c_inv_N[10+m] = 1.0 / N[m]
        c_inv_N[10-m] = 1.0 / N[m]
    inits.append(c_inv_N)
    
    # 5. Alternating signs to cancel peaks?
    c_alt = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_alt[i] = 1.0 if m % 2 == 0 else -1.0
    inits.append(c_alt)
    
    # 6. Gaussian decay in m (Centered at 0)
    for sigma in [1.0, 2.0, 3.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-m**2 / (2 * sigma**2))
        inits.append(c_gauss)
        
    # 7. Gaussian decay in m (Centered at 10)
    for sigma in [1.0, 2.0, 3.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-(m-10)**2 / (2 * sigma**2))
        inits.append(c_gauss)

    # 8. Random symmetric
    np.random.seed(42)
    for _ in range(10):
        c_rand = np.random.randn(dim)
        # Enforce symmetry c[10+m] = c[10-m]
        for m in range(1, 11):
            c_rand[10-m] = c_rand[10+m]
        inits.append(c_rand)
    
    # 9. Anti-symmetric patterns (sin/cos phase shifts)
    # c[10+m] = cos(alpha*m), c[10-m] = sin(alpha*m) type patterns
    for alpha in [np.pi/4, np.pi/3, np.pi/6, np.pi/2]:
        c_anti = np.zeros(dim)
        c_anti[10] = 1.0
        for m in range(1, 11):
            c_anti[10+m] = np.cos(alpha * m)
            c_anti[10-m] = np.sin(alpha * m)
        inits.append(c_anti)
    
    # 10. m-weighted: emphasize mid-m modes
    for weight in [0.5, 1.0, 1.5, 2.0]:
        c_mid = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_mid[i] = np.exp(-m**2 / (2 * weight**2))
        inits.append(c_mid)
    
    # 11. Paired modes: emphasize specific m pairs
    for m_pair in [1, 2, 3, 4, 5, 7, 8]:
        c_pair = np.zeros(dim)
        c_pair[10 + m_pair] = 1.0
        c_pair[10 - m_pair] = 1.0
        inits.append(c_pair)
    
    # 12. Comb patterns
    for period in [2, 3, 4, 5]:
        c_comb = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_comb[i] = np.cos(2 * np.pi * m / period)
        inits.append(c_comb)
        
    return inits

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    dim = 21
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    # Generate diverse initial points
    inits = _generate_symmetric_initial_points(dim)
    
    # Evaluate all inits and keep track of best
    init_vals = []
    for c_init in inits:
        c_norm = _normalize_unit_l2(c_init)
        val = objective(c_norm)
        init_vals.append((val, c_norm))
        if val < best_val:
            best_val = val
            best_coeffs = c_norm.copy()
            
    # Sort initializers by value
    init_vals.sort(key=lambda x: x[0])
    
    # Phase 1: L-BFGS-B optimization from best starts
    n_starts_lgbfsb = 25
    for val, c_start in init_vals[:n_starts_lgbfsb]:
        if time.time() > deadline - 180:
            break
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Gradient descent polishing on the best found so far
    if time.time() < deadline - 60:
        c_polish = best_coeffs.copy()
        val_polish, c_polish_opt = _gradient_descent_polish(c_polish, deadline, max_iter=3000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish_opt.copy()
        
        # Additional L-BFGS-B polish
        if time.time() < deadline - 30:
            try:
                val_polish2, c_polish2 = _lgbfsb_optimize(
                    best_coeffs, deadline, norm_func=_normalize_unit_l2
                )
                if val_polish2 < best_val:
                    best_val = val_polish2
                    best_coeffs = c_polish2.copy()
            except Exception:
                pass
        
        # High-precision final polish
        if time.time() < deadline - 5:
            c_final = best_coeffs.copy()
            val_final, c_final_opt = _high_precision_polish(c_final, deadline)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = c_final_opt.copy()

    # Final validation and normalization
    coeffs = best_coeffs.astype(np.float64)
    
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
        
    max_c = np.max(np.abs(coeffs))
    
    if max_c < 1e-3:
        if max_c > 0:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = np.ones(dim)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
        
    return coeffs

# EVOLVE-BLOCK-END
