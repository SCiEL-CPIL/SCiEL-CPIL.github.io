# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m=-10..10) for degree-10 spherical harmonics
    that minimize R = ||f||_4 / ||f||_2 over the unit sphere S^2.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    # Deadline guard: 630s wall-clock limit, keep safety margin
    deadline = time.time() + 600 - 30  # 30s safety margin
    
    # --- Precompute quadrature and basis ---
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # x_gl = cos(theta), w_gl are the weights
    # Surface measure: dOmega = dphi * d(cos theta), so integral = sum w_i * (sum over phi)
    # Total mass should be 4*pi: sum(w_gl) * (2*pi) = 2 * 2*pi = 4*pi ✓ (since sum of GL weights = 2)
    
    # Equally spaced phi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Build the basis matrix: B[i, j] = Y_j(theta_i, phi_j) for all (theta, phi) pairs
    # But we need to be careful: the quadrature is a tensor product
    # We'll flatten to 1D arrays for efficiency
    
    # cos(theta) values
    ct = x_gl  # shape (n_theta,)
    st = np.sqrt(np.maximum(0, 1 - ct**2))  # sin(theta)
    
    # Precompute associated Legendre polynomials P_10^m(cos theta) for m=0..10
    # Using scipy.special.lpmv which includes Condon-Shortley phase
    # lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase
    # For m > 0: Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi)
    # For m < 0: Y_m = sqrt(2) * N_|m| * P_10^{|m|}(cos theta) * sin(|m|*phi)
    # For m = 0: Y_0 = N_0 * P_10^0(cos theta)
    
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # For m=0: N_0 = sqrt(21/(4*pi) * 10!/10!) = sqrt(21/(4*pi))
    
    def compute_N(m):
        # N_m = sqrt(21/(4*pi) * (10-|m|)!/(10+|m|)!)
        # Use log-gamma for numerical stability
        from math import lgamma, pi
        abs_m = abs(m)
        log_N_sq = np.log(21.0 / (4.0 * np.pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
        return np.sqrt(np.exp(log_N_sq))
    
    # Precompute P_10^m(cos theta) for m = 0..10, shape (n_theta, 11)
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, ct)
    
    # Precompute cos(m*phi) and sin(m*phi) for m = 1..10, shape (n_phi, 10)
    cos_m_phi = np.zeros((n_phi, k))
    sin_m_phi = np.zeros((n_phi, k))
    for m in range(1, k + 1):
        cos_m_phi[:, m-1] = np.cos(m * phi)
        sin_m_phi[:, m-1] = np.sin(m * phi)
    
    # Build the full basis using Kronecker products for efficiency
    n_pts = n_theta * n_phi
    B = np.zeros((n_pts, n_coeffs))
    
    # For m = 0: Y_0 = N_0 * P_10^0(ct) * 1 (constant in phi)
    N_0 = compute_N(0)
    val_theta_0 = N_0 * P[:, 0]  # shape (n_theta,)
    # B for m=0: repeat val_theta_0 for each phi
    B[:, k] = np.tile(val_theta_0, n_phi)
    
    # For m > 0: Y_m = sqrt(2) * N_m * P_10^m(ct) * cos(m*phi)
    # For m < 0: Y_m = sqrt(2) * N_|m| * P_10^{|m|}(ct) * sin(|m|*phi)
    for m in range(1, k + 1):
        N_m = compute_N(m)
        val_theta = np.sqrt(2.0) * N_m * P[:, m]  # shape (n_theta,)
        
        # Positive m: index j = k + m
        B[:, k + m] = np.kron(val_theta, cos_m_phi[:, m-1])
        # Negative m: index j = k - m
        B[:, k - m] = np.kron(val_theta, sin_m_phi[:, m-1])
    
    # Quadrature weights: each point (i, l) has weight w_gl[i] * dphi
    # Total integral: sum_{i,l} f(theta_i, phi_l)^p * w_gl[i] * dphi
    w = np.repeat(w_gl, n_phi) * dphi  # shape (n_pts,)
    
    # --- Objective and gradient ---
    # R = ||f||_4 / ||f||_2
    # ||f||_2^2 = sum w * f^2
    # ||f||_4^4 = sum w * f^4
    # R = (sum w * f^4)^(1/4) / (sum w * f^2)^(1/2)
    
    # We minimize R. Since R is scale-invariant, we can fix ||c||_2 = 1.
    # Actually, let's work on the unit sphere of coefficient space.
    
    def compute_R(c):
        f = B @ c  # shape (n_pts,)
        f2 = f * f
        f4 = f2 * f2
        norm2_sq = np.sum(w * f2)
        norm4_4 = np.sum(w * f4)
        if norm2_sq <= 0:
            return np.inf
        R2 = (norm4_4 / norm2_sq) ** 0.5  # R^2 = (||f||_4^4 / ||f||_2^4) = (norm4_4 / norm2_sq^2)
        # Wait: R = ||f||_4 / ||f||_2
        # R^2 = ||f||_4^2 / ||f||_2^2 = (||f||_4^4)^(1/2) / ||f||_2^2
        # Let me redo:
        # ||f||_2 = (sum w f^2)^(1/2)
        # ||f||_4 = (sum w f^4)^(1/4)
        # R = (sum w f^4)^(1/4) / (sum w f^2)^(1/2)
        # R^2 = (sum w f^4)^(1/2) / (sum w f^2)
        # R^4 = (sum w f^4) / (sum w f^2)^2
        R = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
        return R
    
    def objective_and_grad(c):
        """
        Minimize R^4 = (sum w f^4) / (sum w f^2)^2
        This is equivalent to minimizing R, and R^4 is smoother.
        """
        f = B @ c
        f2 = f * f
        f4 = f2 * f2
        
        S2 = np.sum(w * f2)  # ||f||_2^2
        S4 = np.sum(w * f4)  # ||f||_4^4
        
        if S2 <= 0:
            return np.inf, np.zeros_like(c)
        
        # R^4 = S4 / S2^2
        val = S4 / (S2 * S2)
        
        # Gradient of R^4 with respect to c:
        # d(S4)/dc = 4 * B^T (w * f^3)
        # d(S2)/dc = 2 * B^T (w * f)
        # d(val)/dc = [4 * B^T(w f^3) * S2^2 - S4 * 2 * S2 * 2 * B^T(w f)] / S2^4
        #            = [4 * B^T(w f^3) * S2 - 4 * S4 * B^T(w f)] / S2^3
        #            = 4 * [B^T(w f^3) * S2 - S4 * B^T(w f)] / S2^3
        
        grad_S4 = 4.0 * (B.T @ (w * f**3))
        grad_S2 = 2.0 * (B.T @ (w * f))
        
        grad_val = (grad_S4 * S2 - 2.0 * S4 * grad_S2) / (S2 * S2 * S2)
        
        return val, grad_val
    
    # --- Optimization on the unit sphere ---
    # We'll use a projected gradient / L-BFGS approach.
    # Since the problem is scale-invariant, we fix ||c|| = 1.
    
    from scipy.optimize import minimize
    
    def sphere_objective(c):
        # Project to unit sphere
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.inf
        c_norm = c / norm
        val, _ = objective_and_grad(c_norm)
        return val
    
    def sphere_grad(c):
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.zeros_like(c)
        c_norm = c / norm
        val, grad = objective_and_grad(c_norm)
        # Project gradient onto tangent space
        grad_proj = grad - np.dot(grad, c_norm) * c_norm
        return grad_proj
    
    # Generate diverse initial points on the unit sphere
    rng = np.random.default_rng(42)
    
    best_val = np.inf
    best_c = None
    
    # Try a set of diverse restarts
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point on unit sphere
        c0 = rng.standard_normal(n_coeffs)
        c0 /= np.linalg.norm(c0)
        
        # Also try some structured initial points
        if i == 0:
            c0 = np.ones(n_coeffs) / np.sqrt(n_coeffs)
        elif i == 1:
            # Alternating signs
            c0 = np.ones(n_coeffs)
            c0[::2] = -1.0
            c0 /= np.linalg.norm(c0)
        elif i == 2:
            # Only even m
            c0 = np.zeros(n_coeffs)
            for j in range(0, n_coeffs, 2):
                c0[j] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 3:
            # Only odd m
            c0 = np.zeros(n_coeffs)
            for j in range(1, n_coeffs, 2):
                c0[j] = 1.0
            c0 /= np.linalg.norm(c0)
        elif i == 4:
            # Only m=0
            c0 = np.zeros(n_coeffs)
            c0[k] = 1.0
        
        # Optimize
        try:
            res = minimize(
                sphere_objective,
                c0,
                jac=sphere_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_res = res.x
            norm = np.linalg.norm(c_res)
            if norm > 1e-15:
                c_res /= norm
            
            val_res, _ = objective_and_grad(c_res)
            
            if val_res < best_val:
                best_val = val_res
                best_c = c_res.copy()
                
        except Exception:
            continue
    
    # If no valid result, fall back to all-ones
    if best_c is None:
        best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    # Validate
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    return best_c

# EVOLVE-BLOCK-END
