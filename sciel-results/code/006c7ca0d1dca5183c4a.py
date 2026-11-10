# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, pi

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget with safety margin
    
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    # Build quadrature grid
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1] = cos(theta)
    # weights for integral over [-1,1]
    
    # phi grid
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Build basis matrix: for each (theta_i, phi_j), compute Y_m for m=-10..10
    # Y_0 = N_0 * P_10^0(cos(theta))
    # Y_m = sqrt(2) * N_m * P_10^m(cos(theta)) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos(theta)) * sin(|m|*phi) for m < 0
    
    # Compute associated Legendre polynomials P_10^m(x) for m=0..10
    # Using scipy or manual computation
    from scipy.special import lpmv
    
    # P_10^m(x) for m = 0, 1, ..., 10
    # lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase
    P = np.zeros((n_theta, 11))  # P[i, m] = P_10^m(x_i)
    for m in range(11):
        P[:, m] = lpmv(m, 10, x)
    
    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * pi) * factorial(10 - m) / factorial(10 + m))
    
    # Build basis matrix B of shape (n_theta * n_phi, 21)
    # Index: (i * n_phi + j) -> (theta_i, phi_j)
    # Column m_idx corresponds to m = m_idx - 10
    
    B = np.zeros((n_theta * n_phi, 21))
    
    for i in range(n_theta):
        for j in range(n_phi):
            idx = i * n_phi + j
            phi_j = phi[j]
            
            # m = 0
            B[idx, 10] = N[0] * P[i, 0]
            
            # m > 0
            for m in range(1, 11):
                B[idx, 10 + m] = np.sqrt(2.0) * N[m] * P[i, m] * np.cos(m * phi_j)
                B[idx, 10 - m] = np.sqrt(2.0) * N[m] * P[i, m] * np.sin(m * phi_j)
    
    # Quadrature weights: dOmega = sin(theta) dtheta dphi = -dx dphi (since x = cos(theta), dx = -sin(theta) dtheta)
    # Integral over sphere = integral_0^{2pi} integral_{-1}^{1} f(x, phi) dx dphi
    # Using Gauss-Legendre: sum_i w_i * sum_j f(x_i, phi_j) * dphi
    # So the weight for each (i, j) point is w_i * dphi
    
    quad_weights = np.zeros(n_theta * n_phi)
    for i in range(n_theta):
        for j in range(n_phi):
            idx = i * n_phi + j
            quad_weights[idx] = w[i] * dphi
    
    # Precompute B^T * W (where W is diagonal with quad_weights)
    # For computing f = B @ c, |f|^2, |f|^4, etc.
    
    # Function to compute R = ||f||_4 / ||f||_2
    def compute_R(c):
        f = B @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.sum(quad_weights * f2)
        norm4_sq = np.sum(quad_weights * f4)
        
        if norm2_sq <= 0:
            return 1e10
        
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_sq))  # ||f||_4 = (integral |f|^4)^{1/4}
        
        return norm4 / norm2
    
    # Gradient of R with respect to c
    # R = ||f||_4 / ||f||_2
    # Let A = ||f||_4^4 = sum w_i f_i^4
    # Let B2 = ||f||_2^2 = sum w_i f_i^2
    # R = A^{1/4} / B2^{1/2}
    # dR/dc_m = (1/4) A^{-3/4} * dA/dc_m / B2^{1/2} - A^{1/4} * (1/2) B2^{-3/2} * dB2/dc_m
    # dA/dc_m = 4 * sum w_i f_i^3 * B[i,m]
    # dB2/dc_m = 2 * sum w_i f_i * B[i,m]
    
    def compute_R_and_grad(c):
        f = B @ c
        f2 = f * f
        f3 = f * f2
        f4 = f2 * f2
        
        norm2_sq = np.sum(quad_weights * f2)
        norm4_sq = np.sum(quad_weights * f4)
        
        if norm2_sq <= 0:
            return 1e10, np.zeros(21)
        
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_sq))
        R = norm4 / norm2
        
        # Gradient
        # dA/dc_m = 4 * sum w_i f_i^3 B[i,m]
        # dB2/dc_m = 2 * sum w_i f_i B[i,m]
        
        dA_dc = 4.0 * (B.T @ (quad_weights * f3))
        dB2_dc = 2.0 * (B.T @ (quad_weights * f))
        
        A = norm4_sq
        B2 = norm2_sq
        
        dR_dc = (1.0/4.0) * A**(-3.0/4.0) * dA_dc / np.sqrt(B2) - \
                A**(1.0/4.0) * (1.0/2.0) * B2**(-3.0/2.0) * dB2_dc
        
        return R, dR_dc
    
    # Start with a good initial guess: try single mode first
    # A pure m=10 cosine mode should give R close to 1
    c_best = np.zeros(21)
    c_best[20] = 1.0  # m=10
    R_best = compute_R(c_best)
    
    # Also try m=0
    c_test = np.zeros(21)
    c_test[10] = 1.0
    R_test = compute_R(c_test)
    if R_test < R_best:
        R_best = R_test
        c_best = c_test.copy()
    
    # Try all single modes
    for m_idx in range(21):
        c_test = np.zeros(21)
        c_test[m_idx] = 1.0
        R_test = compute_R(c_test)
        if R_test < R_best:
            R_best = R_test
            c_best = c_test.copy()
    
    # Now do gradient descent from best single mode
    c = c_best.copy()
    
    # L-BFGS-like optimization using scipy
    from scipy.optimize import minimize
    
    def objective(c):
        return compute_R(c)
    
    def objective_grad(c):
        R, grad = compute_R_and_grad(c)
        return R, grad
    
    # Run optimization
    best_c = c_best.copy()
    best_R = R_best
    
    # Try multiple starting points
    start_points = [c_best.copy()]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        c_rand = c_best + 0.1 * rng.standard_normal(21)
        start_points.append(c_rand)
    
    # Also try symmetric combinations
    for m in range(1, 11):
        c_sym = np.zeros(21)
        c_sym[10 + m] = 1.0
        c_sym[10 - m] = 1.0
        start_points.append(c_sym)
        
        c_sym2 = np.zeros(21)
        c_sym2[10 + m] = 1.0
        start_points.append(c_sym2)
    
    for sp in start_points:
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(objective_grad, sp, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_R:
                best_R = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 30:
        try:
            res = minimize(objective_grad, best_c, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_R:
                best_R = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Validate result
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return c_best
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        return c_best
    
    return best_c


# EVOLVE-BLOCK-END
