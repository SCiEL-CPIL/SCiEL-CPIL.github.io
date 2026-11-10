# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

def _precompute_basis():
    """Precompute basis on 70x130 grid and Gaunt coefficients for spectral norm computation."""
    k = 10
    n_theta = 70
    n_phi = 130
    dim = 2 * k + 1  # 21

    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Associated Legendre polynomials P_10^m(x)
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)

    # Normalization N_m = sqrt((2k+1)/(4pi) * (k-m)!/(k+m)!)
    N = np.zeros(11)
    for m in range(11):
        num = factorial(k - m)
        den = factorial(k + m)
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))

    sqrt2 = np.sqrt(2.0)

    # Build B matrix (n_pts x dim)
    n_pts = n_theta * n_phi
    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)

    # m = 0
    val0 = N[0] * P[0, :]
    B[:, 10] = np.repeat(val0, n_phi)

    for m in range(1, 11):
        idx_pos = m + 10
        idx_neg = -m + 10
        val_m = sqrt2 * N[m] * P[m, :]
        cos_term = np.cos(m * phi)
        sin_term = np.sin(m * phi)
        B[:, idx_pos] = (val_m[:, None] * cos_term[None, :]).ravel()
        B[:, idx_neg] = (val_m[:, None] * sin_term[None, :]).ravel()

    # Compute Gaunt coefficients for spectral ||f||_4^4
    # ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l * integral Y_i Y_j Y_k Y_l dOmega
    # = sum_{i,j,k,l} c_i c_j c_k c_l * G_{ijkl}
    # We precompute the 4-index tensor G (21^4 = 194481 entries)
    # Using: Y_i * Y_j is expanded in harmonics, then contracted with Y_k * Y_l
    # For real harmonics, the product Y_i Y_j can be written as sum_m A_{ij}^m Y_m
    # Then integral Y_i Y_j Y_k Y_l = sum_m A_{ij}^m * delta_{mk} * ... 
    # Actually: integral Y_i Y_j Y_k Y_l = sum_{m,n} A_{ij}^m A_{kl}^n * integral Y_m Y_n = sum_m A_{ij}^m A_{kl}^m
    # where A_{ij}^m = integral Y_i Y_j Y_m dOmega (Gaunt coefficient)
    
    # Compute Gaunt coefficients G[i,j,k] = integral Y_i Y_j Y_k dOmega
    # Using the quadrature (exact for degree up to 30, product of three degree-10 harmonics is degree 30)
    G = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for kk in range(dim):
                G[i, j, kk] = np.dot(weights, B[:, i] * B[:, j] * B[:, kk])

    # Now ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l * sum_m G[i,j,m] * G[k,l,m]
    # = sum_m (sum_{i,j} c_i c_j G[i,j,m]) * (sum_{k,l} c_k c_l G[k,l,m])
    # Let S_m(c) = sum_{i,j} c_i c_j G[i,j,m] = c^T G_m c where G_m is the 21x21 matrix G[:,:,m]
    # Then ||f||_4^4 = sum_m S_m(c)^2
    
    # Precompute G_m matrices for fast evaluation
    G_mats = np.zeros((dim, dim, dim))  # G_mats[m, i, j] = G[i, j, m]
    G_mats = G.transpose(2, 0, 1)  # shape (dim, dim, dim): G_mats[m, i, j] = G[i,j,m]

    return B, weights, G_mats

def _compute_ratio_grid(B, weights, c):
    """Compute R = ||f||_4 / ||f||_2 using grid quadrature."""
    f = B @ c
    norm2_sq = np.dot(weights, f**2)
    if norm2_sq < 1e-15:
        return np.inf
    norm4_4 = np.dot(weights, f**4)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def _compute_ratio_spectral(G_mats, c):
    """Compute R using spectral (Gaunt) method. Much faster than grid."""
    # ||f||_2^2 = sum c_m^2 (orthonormal basis)
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-15:
        return np.inf
    # S_m = c^T G_m c for each m
    # G_mats shape: (dim, dim, dim) where G_mats[m, i, j]
    # S_m = sum_{i,j} c_i c_j G_mats[m, i, j]
    # Efficient: S = G_mats @ c @ c^T ... no, S_m = (c ⊗ c)^T vec(G_m)
    # Better: S = np.einsum('mij,i,j->m', G_mats, c, c)
    S = np.einsum('mij,i,j->m', G_mats, c, c)
    norm4_4 = np.dot(S, S)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2

def _compute_grad_spectral(G_mats, c):
    """Gradient of R = ||f||_4 / ||f||_2 w.r.t. c using spectral method."""
    norm2_sq = np.dot(c, c)
    if norm2_sq < 1e-15:
        return np.zeros_like(c)
    
    S = np.einsum('mij,i,j->m', G_mats, c, c)
    norm4_4 = np.dot(S, S)
    
    if norm4_4 < 1e-30:
        return np.zeros_like(c)
    
    # R = norm4_4^{1/4} / norm2_sq^{1/4}
    # dR/dc_i = (1/4) norm4_4^{-3/4} * d(norm4_4)/dc_i * norm2_sq^{-1/4}
    #          - (1/4) norm4_4^{1/4} * d(norm2_sq)/dc_i * norm2_sq^{-5/4}
    # d(norm4_4)/dc_i = 2 * sum_m S_m * dS_m/dc_i
    # dS_m/dc_i = 2 * sum_j G_mats[m, i, j] c_j (since S_m = sum_{i,j} c_i c_j G_{m,ij} and G is symmetric in i,j)
    # Actually S_m = c^T G_m c, so dS_m/dc = 2 G_m c (if G_m symmetric)
    # dS_m/dc_i = 2 * sum_j G_mats[m, i, j] c_j
    
    # d(norm2_sq)/dc_i = 2 c_i
    
    # Let's compute:
    # dS/dc shape: (dim, dim) where dS/dc[m, i] = dS_m/dc_i
    dS_dc = 2.0 * np.einsum('mij,j->mi', G_mats, c)  # shape (dim, dim)
    
    dnorm4_4_dc = 2.0 * np.einsum('m,mi->i', S, dS_dc)  # shape (dim,)
    dnorm2_sq_dc = 2.0 * c
    
    term1 = 0.25 * (norm4_4 ** (-0.75)) * (norm2_sq ** (-0.25)) * dnorm4_4_dc
    term2 = -0.25 * (norm4_4 ** 0.25) * (norm2_sq ** (-1.25)) * dnorm2_sq_dc
    
    return term1 + term2

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm."""
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c
    return c / nrm

def _generate_initial_points():
    """Generate diverse initial points on the unit sphere in R^21."""
    points = []
    
    # All ones
    c = np.ones(21)
    points.append(_normalize_unit_l2(c))
    
    # Only m=0
    c = np.zeros(21); c[10] = 1.0
    points.append(c.copy())
    
    # Only m=10 (cos)
    c = np.zeros(21); c[20] = 1.0
    points.append(c.copy())
    
    # Only m=-10 (sin)
    c = np.zeros(21); c[0] = 1.0
    points.append(c.copy())
    
    # m=0 and m=10
    c = np.zeros(21); c[10] = 1.0; c[20] = 1.0
    points.append(_normalize_unit_l2(c))
    
    # m=0 and m=-10
    c = np.zeros(21); c[10] = 1.0; c[0] = 1.0
    points.append(_normalize_unit_l2(c))
    
    # Alternating: c_m = (-1)^m
    c = np.array([(-1)**(i-10) for i in range(21)], dtype=float)
    points.append(_normalize_unit_l2(c))
    
    # Only even m: m = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 -> indices 0,2,4,6,8,10,12,14,16,18,20
    c = np.zeros(21)
    for idx in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        c[idx] = 1.0
    points.append(_normalize_unit_l2(c))
    
    # Only odd m: m = -9, -7, -5, -3, -1, 1, 3, 5, 7, 9 -> indices 1,3,5,7,9,11,13,15,17,19
    c = np.zeros(21)
    for idx in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        c[idx] = 1.0
    points.append(_normalize_unit_l2(c))
    
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(30):
        c = rng.randn(21)
        points.append(_normalize_unit_l2(c))
    
    # More structured: pairs
    for m in range(1, 11):
        # Only m and -m
        c = np.zeros(21)
        c[m + 10] = 1.0
        c[-m + 10] = 1.0
        points.append(_normalize_unit_l2(c))
        c[m + 10] = 1.0
        c[-m + 10] = -1.0
        points.append(_normalize_unit_l2(c))
    
    # m=0 and pairs
    for m in range(1, 11, 2):
        c = np.zeros(21)
        c[10] = 1.0
        c[m + 10] = 1.0
        c[-m + 10] = 1.0
        points.append(_normalize_unit_l2(c))
    
    return points

def _lgbfsb_optimize(G_mats, c0, maxiter=500):
    """L-BFGS-B optimization of R on the unit sphere (with normalization at each step)."""
    c0 = _normalize_unit_l2(c0.copy())
    
    def objective(c):
        c_norm = _normalize_unit_l2(c)
        return _compute_ratio_spectral(G_mats, c_norm)
    
    def gradient(c):
        c_norm = _normalize_unit_l2(c)
        g = _compute_grad_spectral(G_mats, c_norm)
        # Project gradient onto tangent space of sphere: g - (g·c_norm) c_norm
        g_tangent = g - np.dot(g, c_norm) * c_norm
        return g_tangent
    
    # Use L-BFGS-B with a penalty for deviation from unit norm
    # Actually, let's just optimize on R^21 and normalize at the end
    # The objective is scale-invariant, so we can optimize freely
    
    def obj_free(c):
        return _compute_ratio_spectral(G_mats, c)
    
    def grad_free(c):
        return _compute_grad_spectral(G_mats, c)
    
    res = minimize(
        obj_free,
        c0,
        method='L-BFGS-B',
        jac=grad_free,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    return res.x, res.fun

def _nelder_mead_minimize(G_mats, c0, maxiter=200):
    """Nelder-Mead for polishing."""
    c0 = _normalize_unit_l2(c0.copy())
    
    def objective(c):
        return _compute_ratio_spectral(G_mats, c)
    
    res = minimize(
        objective,
        c0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12}
    )
    return res.x, res.fun

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin (630s limit)

    B, weights, G_mats = _precompute_basis()
    
    # Verify spectral vs grid for a test point
    c_test = np.ones(21) / np.sqrt(21)
    R_grid = _compute_ratio_grid(B, weights, c_test)
    R_spec = _compute_ratio_spectral(G_mats, c_test)
    # They should be very close
    
    best_coeffs = c_test.copy()
    best_R = R_spec
    
    # Generate initial points and evaluate
    init_points = _generate_initial_points()
    for c0 in init_points:
        R0 = _compute_ratio_spectral(G_mats, c0)
        if R0 < best_R:
            best_R = R0
            best_coeffs = c0.copy()
    
    # Polish each initial point with L-BFGS-B
    for c0 in init_points[:25]:  # Limit number of polishes
        if time.time() > deadline - 60:
            break
        try:
            c_polished, R_polished = _lgbfsb_optimize(G_mats, c0, maxiter=300)
            if R_polished < best_R:
                best_R = R_polished
                best_coeffs = c_polished.copy()
        except Exception:
            pass
    
    # Further polish the best with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            c_nm, R_nm = _nelder_mead_minimize(G_mats, best_coeffs, maxiter=200)
            if R_nm < best_R:
                best_R = R_nm
                best_coeffs = c_nm.copy()
        except Exception:
            pass
        
        # Another L-BFGS-B from the Nelder-Mead result
        try:
            c_polished, R_polished = _lgbfsb_optimize(G_mats, best_coeffs, maxiter=500)
            if R_polished < best_R:
                best_R = R_polished
                best_coeffs = c_polished.copy()
        except Exception:
            pass
    
    # Random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        rng = np.random.RandomState(123)
        n_restarts = min(10, int(remaining_time / 10))
        for i in range(n_restarts):
            if time.time() > deadline - 20:
                break
            c0 = rng.randn(21)
            c0 = _normalize_unit_l2(c0)
            try:
                c_polished, R_polished = _lgbfsb_optimize(G_mats, c0, maxiter=200)
                if R_polished < best_R:
                    best_R = R_polished
                    best_coeffs = c_polished.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            c_final, R_final = _lgbfsb_optimize(G_mats, best_coeffs, maxiter=1000)
            if R_final < best_R:
                best_R = R_final
                best_coeffs = c_final.copy()
        except Exception:
            pass
    
    # Validate and ensure valid range
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    
    max_c = np.max(np.abs(best_coeffs))
    if max_c > 1e5:
        best_coeffs = best_coeffs / max_c * 1e5
    if max_c < 1e-3:
        if max_c == 0:
            best_coeffs = np.ones(21)
        else:
            best_coeffs = best_coeffs / max_c * 1e-3
    
    return best_coeffs

# EVOLVE-BLOCK-END
