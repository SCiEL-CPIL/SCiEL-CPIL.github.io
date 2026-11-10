# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Precompute quadrature nodes and weights
_N_GL = 70
_N_PHI = 130
_x_gl, _w_gl = np.polynomial.legendre.leggauss(_N_GL)
_theta_gl = np.arccos(_x_gl)
_phi_grid = 2 * np.pi * np.arange(_N_PHI) / _N_PHI
_dphi = 2 * np.pi / _N_PHI

# Precompute spherical harmonic values at all quadrature points
# Shape: (N_points, 21) for m = -10..10
_k = 10
_N_COEFFS = 2 * _k + 1
_N_POINTS = _N_GL * _N_PHI

# Flatten theta and phi grids for matrix multiplication
# We need Y[i, j] where i is point index, j is m index
# Point index p corresponds to theta[p // _N_PHI] and phi[p % _N_PHI]

# Create meshgrid
_theta_grid, _phi_grid_2d = np.meshgrid(_theta_gl, _phi_grid, indexing='ij')
_theta_flat = _theta_grid.flatten()
_phi_flat = _phi_grid_2d.flatten()
_cos_theta_flat = _x_gl[:, None] * np.ones((1, _N_PHI))
_cos_theta_flat = _cos_theta_flat.flatten()

# Weights
_W_flat = np.tile(_w_gl, _N_PHI) * _dphi

# Y matrix: (N_POINTS, _N_COEFFS)
_Y_matrix = np.zeros((_N_POINTS, _N_COEFFS))

for idx, m in enumerate(range(-_k, _k + 1)):
    abs_m = abs(m)
    P = lpmv(abs_m, _k, _x_gl)  # shape (70,)
    P_flat = np.tile(P, _N_PHI)
    
    n_fact = factorial(_k - abs_m, exact=True)
    d_fact = factorial(_k + abs_m, exact=True)
    N_m = np.sqrt((2 * _k + 1) / (4 * np.pi) * (n_fact / d_fact))
    
    if m == 0:
        _Y_matrix[:, idx] = N_m * P_flat
    elif m > 0:
        cos_term = np.cos(m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * cos_term
    else:
        sin_term = np.sin(abs_m * _phi_flat)
        _Y_matrix[:, idx] = np.sqrt(2) * N_m * P_flat * sin_term

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature.
    c: length 21 vector
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf
    
    f_val = _Y_matrix @ c
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _compute_ratio_gradient(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    Returns (ratio, gradient)
    """
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return np.inf, np.zeros_like(c)
    
    f_val = _Y_matrix @ c
    f2 = f_val ** 2
    f4 = f_val ** 4
    
    norm2_sq = np.dot(f2, _W_flat)
    norm4_4 = np.dot(f4, _W_flat)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    ratio = norm4 / norm2
    
    # Gradient of ||f||_4^4 = sum w_i f_i^4 -> 4 * sum w_i f_i^3 * Y_i
    # d/dc ||f||_4^4 = 4 * Y^T (w * f^3)
    # d/dc ||f||_2^2 = 2 * Y^T (w * f)
    # d/dc ratio = d/dc (norm4/norm2) = (1/norm2) * d(norm4)/dc - (norm4/norm2^2) * d(norm2)/dc
    # d(norm4)/dc = (1/(4*norm4^3)) * 4 * Y^T(w*f^3) = Y^T(w*f^3)/norm4^3
    # d(norm2)/dc = (1/(2*norm2)) * 2 * Y^T(w*f) = Y^T(w*f)/norm2
    
    grad_norm4_4 = _Y_matrix.T @ (_W_flat * f_val ** 3)
    grad_norm2_sq = _Y_matrix.T @ (_W_flat * f_val)
    
    grad_norm4 = grad_norm4_4 / (4 * norm4 ** 3)
    grad_norm2 = grad_norm2_sq / (2 * norm2)
    
    gradient = (1 / norm2) * grad_norm4 - (norm4 / norm2 ** 2) * grad_norm2
    
    return ratio, gradient

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 600  # seconds, with safety margin
    
    best_coeffs = None
    best_ratio = np.inf
    
    def _update_best(c, r):
        nonlocal best_coeffs, best_ratio
        if r < best_ratio and np.all(np.isfinite(c)):
            best_ratio = r
            best_coeffs = c.copy()
    
    n_coeffs = _N_COEFFS
    rng = np.random.default_rng(42)
    
    # 1. Initial structured guesses
    candidates = []
    
    # All ones
    candidates.append(np.ones(n_coeffs))
    
    # Single modes
    for i in range(n_coeffs):
        c = np.zeros(n_coeffs)
        c[i] = 1.0
        candidates.append(c)
    
    # Pairs of modes (cos/sin symmetry often helps)
    for i in range(n_coeffs):
        for j in range(i+1, n_coeffs):
            c = np.zeros(n_coeffs)
            c[i] = 1.0
            c[j] = 1.0
            candidates.append(c)
    
    # Sparse random patterns
    for _ in range(30):
        n_nonzero = rng.integers(3, 12)
        indices = rng.choice(n_coeffs, size=n_nonzero, replace=False)
        c = np.zeros(n_coeffs)
        c[indices] = rng.normal(size=n_nonzero)
        candidates.append(c)
        
    # Symmetric patterns: even m only
    c_even = np.zeros(n_coeffs)
    c_even[::2] = 1.0
    candidates.append(c_even)
    
    # Odd m only
    c_odd = np.zeros(n_coeffs)
    c_odd[1::2] = 1.0
    candidates.append(c_odd)
    
    # Alternating signs
    c_alt = np.ones(n_coeffs)
    c_alt[1::2] = -1.0
    candidates.append(c_alt)
    
    # Additional structured patterns: high-|m| dominance
    for target_m in [4, 5, 6, 7, 8, 9, 10]:
        c = np.zeros(n_coeffs)
        c[target_m - 10] = 1.0  # cos mode
        c[-target_m - 10] = 1.0  # sin mode
        candidates.append(c)
        
    # Mixed patterns with varying weights
    for _ in range(20):
        c = rng.normal(size=n_coeffs)
        # Emphasize certain |m| values
        m_vals = np.arange(-10, 11)
        weight = np.exp(-0.5 * (np.abs(m_vals) - 6) ** 2)
        c *= weight
        candidates.append(c)
    
    # 2. Evaluate all candidates
    for c in candidates:
        r = _compute_ratio(c)
        _update_best(c, r)
        
    # 3. Local optimization from best candidates using analytic gradient
    # Sort candidates by ratio
    scored_candidates = []
    for c in candidates:
        r = _compute_ratio(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 15 and run L-BFGS-B with gradient
    for i, (r_init, c_init) in enumerate(scored_candidates[:15]):
        if time.time() > deadline - 90:
            break
        
        try:
            def _obj_and_grad(x):
                val, grad = _compute_ratio_gradient(x)
                return val, grad
            
            res = minimize(
                _obj_and_grad,
                c_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            c_opt = res.x
            r_opt, _ = _compute_ratio_gradient(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
    
    # 4. Random restarts with optimization using gradient
    n_restarts = 80
    for i in range(n_restarts):
        if time.time() > deadline - 45:
            break
            
        c_start = rng.normal(size=n_coeffs)
        norm_c = np.linalg.norm(c_start)
        if norm_c < 1e-15:
            continue
        c_start_norm = c_start / norm_c
        
        try:
            def _obj_and_grad(x):
                val, grad = _compute_ratio_gradient(x)
                return val, grad
            
            res = minimize(
                _obj_and_grad,
                c_start_norm,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            c_opt = res.x
            r_opt, _ = _compute_ratio_gradient(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass
            
    # 5. Final polish: run L-BFGS-B on the best result with very tight tolerances
    if best_coeffs is not None:
        try:
            def _obj_and_grad(x):
                val, grad = _compute_ratio_gradient(x)
                return val, grad
            
            res = minimize(
                _obj_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            c_opt = res.x
            r_opt, _ = _compute_ratio_gradient(c_opt)
            if np.isfinite(r_opt):
                _update_best(c_opt, r_opt)
        except Exception:
            pass

    # Final validation
    if best_coeffs is None or np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            best_coeffs = np.ones(n_coeffs)
        else:
            best_coeffs = best_coeffs / max_abs * 1.0
            if np.max(np.abs(best_coeffs)) < 1e-3:
                best_coeffs = np.ones(n_coeffs)
    
    return best_coeffs.astype(np.float64)

# EVOLVE-BLOCK-END
