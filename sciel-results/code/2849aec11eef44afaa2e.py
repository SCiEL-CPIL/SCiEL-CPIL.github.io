import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import minimize
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        P = lpmv(abs_m, k, x_gl)
        
        if m == 0:
            vals = N_m * P
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient."""
    f = Y @ coeffs
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10, np.zeros_like(coeffs)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    R = norm4 / norm2
    
    # Gradient: dR/dc = (1/(N2*N4^3)) Y^T (w f^3) - (R/N2^2) Y^T (w f)
    term1 = Y.T @ (weights * f**3)
    term2 = Y.T @ (weights * f)
    grad = (1.0 / (norm2 * norm4**3)) * term1 - (R / norm2**2) * term2
    return R, grad

def _generate_monomial_seed(rng, Y, weights, k=10):
    """
    Generate a seed based on the real part of a rotated complex monomial z^k.
    This exploits the high symmetry of the monomial to start near a 'flat' extremal candidate.
    """
    # Generate a random rotation matrix
    # Using QR decomposition of a random matrix to get a uniform rotation
    A = rng.normal(size=(3, 3))
    Q, R_ = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.sign(np.linalg.det(Q))
    if d < 0:
        Q[:, 0] *= -1
    R = Q.T # Rotation matrix acting on coordinates
    
    # The complex monomial z^k in spherical coordinates is related to Y_k^k.
    # Specifically, Y_k^k (complex) is proportional to sin^k(theta) e^{ik phi}.
    # The real part is sin^k(theta) cos(k phi).
    # In our real basis, Y_k^k (m=k) corresponds to sqrt(2) N_k P_k^k cos(k phi).
    # P_k^k(cos theta) = (-1)^k (2k-1)!! sin^k(theta).
    # So Y_k^k (real basis, m=k) is proportional to sin^k(theta) cos(k phi).
    
    # We want to rotate this function.
    # Instead of rotating the function directly (which is hard in coefficient space),
    # we can sample the rotated function on the quadrature grid and project back to the basis.
    
    # Get quadrature points in Cartesian coordinates
    x_gl, w_gl = np.polynomial.legendre.leggauss(70) # Rebuild or pass in? Let's pass in.
    # Wait, I need the actual theta and phi.
    # x_gl = cos(theta). theta = arccos(x_gl).
    # phi is already defined.
    
    # This function needs x_gl and phi.
    # Let's refactor to pass them.
    pass # Placeholder, will implement in rudin_problem10 where data is available

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    # Precompute Cartesian coordinates for quadrature points to generate monomial seeds
    theta = np.arccos(x_gl)
    # Create meshgrid for theta and phi
    # theta is (n_gl,), phi is (n_phi,)
    # We need to evaluate the rotated monomial on these points.
    
    # The monomial z^k = (x + i y)^k.
    # In spherical: z = r sin(theta) e^{i phi}.
    # z^k = r^k sin^k(theta) e^{i k phi}.
    # Re(z^k) = sin^k(theta) cos(k phi).
    
    # We rotate the coordinates (x,y,z) by a random rotation R.
    # x' = R x, etc.
    # Then we evaluate Re((x' + i y')^k) at the original quadrature points?
    # No, rotation invariance means:
    # f(R^{-1} x) = (R f)(x).
    # If we start with f_0(x) = Re(z^k) = sin^k(theta) cos(k phi),
    # and rotate it by R, we get f_R(x) = f_0(R^{-1} x).
    
    # So, for each seed:
    # 1. Generate random rotation R.
    # 2. For each quadrature point p_i, compute p'_i = R^{-1} p_i.
    # 3. Evaluate f_0(p'_i) = sin^k(theta'_i) cos(k phi'_i).
    # 4. Project this vector of values onto the basis Y to get coefficients.
    
    # Precompute quadrature points in Cartesian
    # x = cos(theta) = x_gl
    # y = sin(theta) cos(phi)
    # z = sin(theta) sin(phi)
    # Note: Our phi is the azimuthal angle.
    # Standard spherical: x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta).
    # But in our Y_m definition:
    # Y_m uses P(cos theta) and cos/sin(m phi).
    # This matches the standard physics convention where theta is polar (from z-axis) and phi is azimuthal (from x-axis).
    # So:
    # x_cart = sin(theta) cos(phi)
    # y_cart = sin(theta) sin(phi)
    # z_cart = cos(theta)
    
    # Let's build the Cartesian grid
    # theta is (n_gl,), phi is (n_phi,)
    # We need a meshgrid to evaluate for all points.
    # But we can be smart.
    # The value at point (theta_i, phi_j) is sin^k(theta_i) cos(k (phi_j + delta_phi_i))? No.
    # Rotation mixes theta and phi.
    
    # Let's just compute the Cartesian coordinates for all points.
    # n_points = n_gl * n_phi = 70 * 130 = 9100. This is small.
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x_cart = np.sin(theta_grid) * np.cos(phi_grid).ravel()
    y_cart = np.sin(theta_grid) * np.sin(phi_grid).ravel()
    z_cart = np.cos(theta_grid).ravel()
    
    # Points matrix: (n_points, 3)
    points = np.column_stack((x_cart, y_cart, z_cart))
    
    n_seeds = 20
    
    for i in range(n_seeds):
        if time.time() > deadline - 60.0:
            break
            
        # Generate random rotation
        A = rng.normal(size=(3, 3))
        Q, R_ = np.linalg.qr(A)
        d = np.sign(np.linalg.det(Q))
        if d < 0:
            Q[:, 0] *= -1
        R_rot = Q.T # Rotation matrix
        
        # Apply inverse rotation to points: p' = R^{-1} p = R^T p
        # points is (N, 3). R_rot is (3, 3).
        # p'_i = R_rot.T @ p_i
        # Vectorized: p' = points @ R_rot
        points_rot = points @ R_rot
        
        # Extract theta' and phi' from rotated points
        x_p = points_rot[:, 0]
        y_p = points_rot[:, 1]
        z_p = points_rot[:, 2]
        
        # r = 1
        sin_theta_p = np.sqrt(x_p**2 + y_p**2)
        # Avoid division by zero
        sin_theta_p[sin_theta_p < 1e-15] = 1e-15
        cos_theta_p = z_p
        
        theta_p = np.arctan2(sin_theta_p, cos_theta_p)
        phi_p = np.arctan2(y_p, x_p)
        
        # Evaluate f_0(p') = sin^k(theta') cos(k phi')
        # Note: The standard Y_k^k is proportional to sin^k(theta) cos(k phi).
        # So this is the correct function to evaluate.
        vals = (np.sin(theta_p) ** k) * np.cos(k * phi_p)
        
        # Project onto basis Y
        # coeffs = (Y^T W Y)^{-1} Y^T W vals
        # Since Y are orthonormal (with respect to the quadrature weights),
        # Y^T W Y = I.
        # So coeffs = Y^T W vals.
        coeffs = Y.T @ (weights * vals)
        
        # Normalize to unit L2 norm (optional, but helps with optimization)
        norm_c = np.linalg.norm(coeffs)
        if norm_c > 1e-10:
            coeffs = coeffs / norm_c
            
        # Refine with L-BFGS-B
        def objective(c):
            val, _ = _compute_ratio_and_grad(c, Y, weights)
            return val
        
        def gradient(c):
            _, g = _compute_ratio_and_grad(c, Y, weights)
            return g
        
        try:
            res = minimize(objective, coeffs, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass
        
        # Also try a few power iteration steps from this seed to see if it improves
        # But L-BFGS-B is usually better for this smooth landscape.
        
    # Fallback if no seeds worked
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0 # m=0
        
    # Final validation and scaling
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            best_coeffs[10] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1.0

    return best_coeffs
