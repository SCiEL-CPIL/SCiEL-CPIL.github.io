# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv


def _compute_R(c, theta_grid, phi_grid, sin_theta, Y_matrix):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed quadrature weights and basis values.
    
    Parameters:
        c: (21,) coefficients for m=-10..10
        theta_grid: (70,) cos-theta values (x_i)
        phi_grid: (130,) phi values
        sin_theta: (70,) sin(theta_i) = sqrt(1 - x_i^2)
        Y_matrix: (70, 130, 21) precomputed Y_m(x_i, phi_j)
    
    Returns:
        R: float, the ratio ||f||_4 / ||f||_2
    """
    # f(x_i, phi_j) = sum_m c_m Y_m(x_i, phi_j)
    # Y_matrix shape: (70, 130, 21)
    f = np.tensordot(c, Y_matrix, axes=([0], [2]))  # (70, 130)
    
    # Quadrature weights
    # For Gauss-Legendre in x = cos(theta):
    #   integral over sphere = sum_i w_i * (1/2) * (1/2*pi) * sum_j f(x_i, phi_j) * sin(theta_i)
    # Wait, let me be more careful.
    # 
    # Surface measure: dOmega = sin(theta) dtheta dphi
    # Let x = cos(theta), dx = -sin(theta) dtheta
    # So sin(theta) dtheta = -dx, and integral becomes integral over x from -1 to 1 of dx * integral over phi
    # 
    # Gauss-Legendre quadrature on [-1, 1]: sum_i w_i g(x_i) approximates integral_{-1}^{1} g(x) dx
    # So integral over sphere = sum_i w_i * integral_{0}^{2pi} f(x_i, phi) dphi
    # 
    # For phi, we use 130 equally spaced points: phi_j = j * 2*pi/130, j=0..129
    # Trapezoidal rule: integral_{0}^{2pi} g(phi) dphi ≈ (2*pi/130) * sum_j g(phi_j)
    # 
    # So: ||f||_p^p ≈ sum_i w_i * (2*pi/130) * sum_j |f(x_i, phi_j)|^p
    # 
    # But wait, the problem says "70-point Gauss-Legendre quadrature in cos theta times 130 equally spaced phi"
    # This should be exact for degree 10 harmonics if the quadrature is chosen appropriately.
    # 
    # Actually, for the product quadrature to be exact for |f|^p where f is degree 10,
    # we need the quadrature to handle products of degree up to 2p*10 in each variable.
    # But the problem states it's exact for this degree, so I'll trust that.
    
    # Get GL weights
    # We need to compute them or have them precomputed. Let me compute them here for clarity,
    # but in the actual program we'll precompute.
    
    # Actually, let me restructure: I'll precompute everything including weights.
    # For now, this is a placeholder that assumes weights are passed or computed.
    
    # Let me just compute R directly with proper weighting.
    # I'll assume the caller has set up the quadrature correctly.
    
    # For simplicity in this helper, let me just compute unweighted sums and assume
    # the quadrature is normalized such that sum of weights * (2pi/130) * sin_theta_i = 4*pi
    # 
    # Actually, let me be more explicit. I'll return the raw integrals and let the caller
    # handle normalization.
    
    # ||f||_2^2 = integral |f|^2 dOmega
    # ||f||_4^4 = integral |f|^4 dOmega
    
    # Using the quadrature:
    # integral ≈ sum_i w_i * (2*pi/N_phi) * sum_j |f(x_i, phi_j)|^p
    # where w_i are GL weights on [-1,1], N_phi = 130
    
    # But I don't have w_i here. Let me restructure the whole thing.
    
    # For now, return f so the caller can compute norms.
    return f


def _precompute_basis():
    """
    Precompute the quadrature grid and spherical harmonic basis values.
    
    Returns:
        theta_grid: (70,) x = cos(theta) values
        phi_grid: (130,) phi values  
        w_theta: (70,) Gauss-Legendre weights
        Y_matrix: (70, 130, 21) Y_m(x_i, phi_j) for m=-10..10
        N_phi: int, number of phi points
    """
    k = 10
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    theta_grid, w_theta = np.polynomial.legendre.leggauss(N_theta)
    
    # Equally spaced phi
    phi_grid = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # sin(theta) = sqrt(1 - cos^2(theta)) = sqrt(1 - x^2)
    sin_theta = np.sqrt(np.maximum(0, 1 - theta_grid**2))
    
    # Precompute Y_m(x_i, phi_j) for all m, i, j
    # Y_0 = N_0 P_10^0(x)
    # Y_m = sqrt(2) N_m P_10^m(x) cos(m*phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(x) sin(|m|*phi) for m < 0
    
    Y_matrix = np.zeros((N_theta, N_phi, 2 * k + 1), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        
        # Condon-Shortley phase is included in scipy's lpmv
        # lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase
        P = lpmv(abs_m, k, theta_grid)  # (N_theta,)
        
        # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Note: 21 = 2*(2*10+1) = 2*21... wait, 2k+1 = 21
        # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
        # For k=10: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        
        # Compute factorial ratio
        # (10-m)!/(10+m)! for m >= 0
        # This can be computed as 1/((10+m)*(10+m-1)*...*(10-m+1))
        if abs_m == 0:
            fact_ratio = 1.0
        else:
            # (10-abs_m)! / (10+abs_m)! = 1 / prod_{j=10-abs_m+1}^{10+abs_m} j
            num_terms = 2 * abs_m
            prod = 1.0
            for j in range(10 - abs_m + 1, 10 + abs_m + 1):
                prod *= j
            fact_ratio = 1.0 / prod
        
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact_ratio)
        
        if m == 0:
            # Y_0 = N_0 P_10^0(x)
            Y_vals = N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * np.ones((1, N_phi))[np.newaxis, :]
        elif m > 0:
            # Y_m = sqrt(2) N_m P_10^m(x) cos(m*phi)
            cos_term = np.cos(m * phi_grid)  # (N_phi,)
            Y_vals = np.sqrt(2.0) * N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * cos_term[np.newaxis, :]
        else:
            # m < 0, |m| = -m
            # Y_m = sqrt(2) N_|m| P_10^|m|(x) sin(|m|*phi)
            sin_term = np.sin(abs_m * phi_grid)  # (N_phi,)
            Y_vals = np.sqrt(2.0) * N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * sin_term[np.newaxis, :]
    
    return theta_grid, phi_grid, w_theta, sin_theta, Y_matrix, N_phi


def _compute_R_from_precomputed(c, w_theta, Y_matrix, N_phi):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed data.
    
    The quadrature is:
    integral g(x, phi) dOmega ≈ sum_i w_i * (2*pi/N_phi) * sum_j g(x_i, phi_j)
    
    where w_i are GL weights on [-1,1].
    
    This is because:
    dOmega = sin(theta) dtheta dphi = dx dphi (with x = cos(theta), dx = -sin(theta)dtheta)
    integral_{-1}^{1} dx integral_0^{2pi} dphi g(x, phi)
    ≈ sum_i w_i * (2*pi/N_phi) * sum_j g(x_i, phi_j)
    """
    # f(x_i, phi_j) = sum_m c_m Y_m(x_i, phi_j)
    f = np.tensordot(c, Y_matrix, axes=([0], [2]))  # (N_theta, N_phi)
    
    # Quadrature weight for each (i, j) point
    # weight_ij = w_i * (2*pi/N_phi)
    # But we can factor this:
    # ||f||_p^p ≈ (2*pi/N_phi) * sum_i w_i * sum_j |f(x_i, phi_j)|^p
    
    f2 = f ** 2
    f4 = f ** 4
    
    # Sum over phi first
    sum_j_f2 = np.sum(f2, axis=1)  # (N_theta,)
    sum_j_f4 = np.sum(f4, axis=1)  # (N_theta,)
    
    # Then apply theta weights
    integral_f2 = np.sum(w_theta * sum_j_f2)
    integral_f4 = np.sum(w_theta * sum_j_f4)
    
    # Multiply by 2*pi/N_phi
    factor = 2.0 * np.pi / N_phi
    norm2_sq = factor * integral_f2
    norm4_4 = factor * integral_f4
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # ||f||_4 = (||f||_4^4)^(1/4)
    
    R = norm4 / norm2
    return R


def _compute_grad_R(c, w_theta, Y_matrix, N_phi):
    """
    Compute the gradient of R with respect to c.
    
    R = ||f||_4 / ||f||_2
    
    dR/dc_m = (1/||f||_2) * d||f||_4/dc_m - (||f||_4/||f||_2^2) * d||f||_2/dc_m
    
    d||f||_4/dc_m = (1/4) ||f||_4^{-3} * d||f||_4^4/dc_m
    d||f||_4^4/dc_m = 4 * integral f^3 Y_m dOmega
    
    d||f||_2/dc_m = (1/2) ||f||_2^{-1} * d||f||_2^2/dc_m
    d||f||_2^2/dc_m = 2 * integral f Y_m dOmega
    
    So:
    dR/dc_m = (1/||f||_2) * (1/4) ||f||_4^{-3} * 4 * integral f^3 Y_m dOmega
             - (||f||_4/||f||_2^2) * (1/2) ||f||_2^{-1} * 2 * integral f Y_m dOmega
           
           = (1/||f||_2) * ||f||_4^{-3} * integral f^3 Y_m dOmega
             - (||f||_4/||f||_2^3) * integral f Y_m dOmega
    """
    f = np.tensordot(c, Y_matrix, axes=([0], [2]))  # (N_theta, N_phi)
    
    factor = 2.0 * np.pi / N_phi
    
    # Compute integrals
    f2 = f ** 2
    f3 = f ** 3
    f4 = f ** 4
    
    # integral f Y_m dOmega for all m
    # integral f Y_m = factor * sum_i w_i * sum_j f(x_i, phi_j) Y_m(x_i, phi_j)
    # = factor * sum_i w_i * sum_j f_ij Y_m_ij
    # 
    # For each m: integral_f_Ym = factor * sum_i w_i * dot(f[:, :], Y_matrix[:, :, m])
    # 
    # Let's compute: inner_ij = f * Y_matrix  -> (N_theta, N_phi, 21)
    # Then sum over j, then weighted sum over i
    
    # More efficiently:
    # integral_f_Ym[m] = factor * sum_i w_i * sum_j f_ij * Y_m_ij
    # = factor * sum_i w_i * <f_i, Y_m_i> where f_i is the phi-slice
    
    # Let me compute this as:
    # temp = f[:, :, np.newaxis] * Y_matrix  # (N_theta, N_phi, 21)
    # sum_j = np.sum(temp, axis=1)  # (N_theta, 21)
    # integral_f_Ym = factor * np.sum(w_theta[:, np.newaxis] * sum_j, axis=0)  # (21,)
    
    temp_fY = f[:, :, np.newaxis] * Y_matrix  # (N_theta, N_phi, 21)
    sum_j_fY = np.sum(temp_fY, axis=1)  # (N_theta, 21)
    integral_f_Ym = factor * np.sum(w_theta[:, np.newaxis] * sum_j_fY, axis=0)  # (21,)
    
    # integral f^3 Y_m dOmega
    temp_f3Y = f3[:, :, np.newaxis] * Y_matrix  # (N_theta, N_phi, 21)
    sum_j_f3Y = np.sum(temp_f3Y, axis=1)  # (N_theta, 21)
    integral_f3_Ym = factor * np.sum(w_theta[:, np.newaxis] * sum_j_f3Y, axis=0)  # (21,)
    
    # Compute norms
    sum_j_f2 = np.sum(f2, axis=1)
    sum_j_f4 = np.sum(f4, axis=1)
    integral_f2 = np.sum(w_theta * sum_j_f2)
    integral_f4 = np.sum(w_theta * sum_j_f4)
    
    norm2_sq = factor * integral_f2
    norm4_4 = factor * integral_f4
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    # dR/dc_m = (1/norm2) * norm4^{-3} * integral_f3_Ym[m] - (norm4/norm2^3) * integral_f_Ym[m]
    coeff1 = 1.0 / (norm2 * norm4**3)
    coeff2 = norm4 / (norm2**3)
    
    grad = coeff1 * integral_f3_Ym - coeff2 * integral_f_Ym
    
    return grad


def _optimize_coeffs(c0, w_theta, Y_matrix, N_phi, max_iter=10000, tol=1e-12):
    """
    Optimize coefficients to minimize R = ||f||_4 / ||f||_2.
    
    Uses gradient descent with adaptive step size and normalization invariance handling.
    """
    c = c0.copy().astype(np.float64)
    c = c / np.linalg.norm(c)  # Normalize initially
    
    best_R = _compute_R_from_precomputed(c, w_theta, Y_matrix, N_phi)
    best_c = c.copy()
    
    step_size = 0.01
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        R = _compute_R_from_precomputed(c, w_theta, Y_matrix, N_phi)
        
        if R < best_R:
            best_R = R
            best_c = c.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        if R - best_R < tol:
            break
        
        grad = _compute_grad_R(c, w_theta, Y_matrix, N_phi)
        
        # Check gradient norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Normalize gradient
        grad = grad / grad_norm
        
        # Try to take a step
        improved = False
        for trial in range(10):
            c_new = c - step_size * grad
            # Re-normalize to handle scaling invariance
            c_new = c_new / np.linalg.norm(c_new)
            
            R_new = _compute_R_from_precomputed(c_new, w_theta, Y_matrix, N_phi)
            
            if R_new < R:
                c = c_new
                improved = True
                break
            else:
                step_size *= 0.5
        
        if not improved:
            # If no improvement, try smaller steps
            step_size *= 0.1
            if step_size < 1e-10:
                break
        
    return best_c, best_R


def _random_start():
    """Generate a random starting point."""
    k = 10
    c = np.random.randn(2 * k + 1)
    c = c / np.linalg.norm(c)
    return c


def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Precompute basis
    theta_grid, phi_grid, w_theta, sin_theta, Y_matrix, N_phi = _precompute_basis()
    
    # Try multiple random starts and keep the best
    np.random.seed(42)
    
    best_c = None
    best_R = np.inf
    
    n_starts = 5
    
    for start in range(n_starts):
        c0 = _random_start() if start > 0 else np.ones(21) / np.sqrt(21.0)
        
        c_opt, R_opt = _optimize_coeffs(c0, w_theta, Y_matrix, N_phi, max_iter=5000, tol=1e-12)
        
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt
    
    # Scale the result to be within valid range
    # Valid: max|c| >= 1e-3 and max|c| <= 1e5
    # Our optimization works on normalized vectors, so let's scale to a reasonable value
    c = best_c.copy()
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    # Final validation
    if len(c) != 21:
        c = np.ones(21)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        c = np.ones(21)
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    return c


# EVOLVE-BLOCK-END
