# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv


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
    k = 10
    start_time = time.time()
    deadline = start_time + 600.0  # Leave margin before 630s kill

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl in [-1, 1], w_gl are weights
    # cos(theta) = x_gl, so dOmega = dphi * d(cos theta)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))  # cos(m*phi) for m=0..10
    sin_m_phi = np.zeros((11, n_phi))  # sin(m*phi) for m=0..10
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    # P^m = P_10^m(cos theta) with Condon-Shortley phase
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        # lpmv(m, l, x) computes P_l^m(x) with Condon-Shortley phase
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the matrix Y such that f(theta, phi) = sum_m c_m Y_m
    # Y is a function of theta (via x_gl) and phi
    # For each quadrature point (i, j) where i is theta index, j is phi index:
    # Y_0 = N_0 * P_10^0(x_i)
    # Y_m = sqrt(2) * N_m * P_10^m(x_i) * cos(m*phi_j) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^{|m|}(x_i) * sin(|m|*phi_j) for m < 0
    
    # We'll evaluate f on the grid: f[i, j] for theta index i, phi index j
    # f = sum_{m=-10}^{10} c_m Y_m
    # Let's build a matrix A where A[i, j, m] = Y_m(theta_i, phi_j)
    # But that's 70 * 130 * 21 = 188,600 elements, manageable
    
    # Actually, let's think of it differently. For fixed theta_i, 
    # f(theta_i, phi_j) = N_0 * P_10^0(x_i) * c_0 
    #                     + sum_{m=1}^{10} sqrt(2) * N_m * P_10^m(x_i) * [c_m * cos(m*phi_j) + c_{-m} * sin(m*phi_j)]
    
    # Let's precompute for each theta_i:
    # base_i = N_0 * P_10^0(x_i)
    # coeff_m_i = sqrt(2) * N_m * P_10^m(x_i) for m=1..10
    
    base = N[0] * P_legendre[0]  # shape (70,)
    coeff_m = np.zeros((10, 70))  # coeff_m[m-1, i] = sqrt(2) * N_m * P_10^m(x_i)
    for m in range(1, 11):
        coeff_m[m-1] = np.sqrt(2.0) * N[m] * P_legendre[m]
    
    # Now f[i, j] = base[i] * c_0 + sum_{m=1}^{10} coeff_m[m-1, i] * (c_m * cos_m_phi[m, j] + c_{-m} * sin_m_phi[m, j])
    
    # Let's define the coefficient vector c in order m = -10, -9, ..., -1, 0, 1, ..., 9, 10
    # Index: m = i - 10, so c[0] = c_{-10}, c[10] = c_0, c[20] = c_{10}
    
    # For optimization, we'll use gradient descent or a similar method.
    # Let's use a simple iterative approach: adjust coefficients to minimize R = ||f||_4 / ||f||_2
    
    # Initial guess: all ones
    c = np.ones(21, dtype=np.float64)
    best_c = c.copy()
    best_R = np.inf
    
    def compute_R(c_vec):
        """Compute ||f||_4 / ||f||_2 for given coefficients."""
        # c_vec: shape (21,), c_vec[i] corresponds to m = i - 10
        c0 = c_vec[10]  # m = 0
        # c_m for m > 0: c_vec[10 + m]
        # c_m for m < 0: c_vec[10 + m] where m is negative, so c_vec[10 + m] = c_vec[10 - |m|]
        
        # f[i, j] for i in 0..69 (theta), j in 0..129 (phi)
        f = np.zeros((70, 130), dtype=np.float64)
        
        # Add m=0 term
        f += base[:, np.newaxis] * c0
        
        # Add m > 0 and m < 0 terms
        for m in range(1, 11):
            c_pos = c_vec[10 + m]  # coefficient for +m
            c_neg = c_vec[10 - m]  # coefficient for -m
            # Term: coeff_m[m-1, i] * (c_pos * cos(m*phi_j) + c_neg * sin(m*phi_j))
            f += coeff_m[m-1, :, np.newaxis] * (c_pos * cos_m_phi[m, np.newaxis, :] + c_neg * sin_m_phi[m, np.newaxis, :])
        
        # Compute norms
        # dOmega = dphi * d(cos theta) = dphi * w_gl[i] * dphi
        # Wait, the quadrature: integral over sphere = sum_i sum_j f(i,j) * w_gl[i] * dphi
        # Because dOmega = dphi * d(cos theta), and Gauss-Legendre approximates integral over cos theta
        
        weights = np.outer(w_gl, np.full(n_phi, dphi))  # shape (70, 130)
        
        f2 = f ** 2
        f4 = f ** 4
        
        norm2_sq = np.sum(f2 * weights)  # ||f||_2^2
        norm4 = np.sum(f4 * weights)     # ||f||_4^4
        
        if norm2_sq <= 0:
            return np.inf
        if norm4 <= 0:
            return 0.0
            
        norm2 = np.sqrt(norm2_sq)
        norm4_val = np.sqrt(np.sqrt(norm4))  # ||f||_4 = (||f||_4^4)^(1/4)
        
        return norm4_val / norm2
    
    # Compute initial R
    best_R = compute_R(c)
    
    # Use gradient-free optimization: try small perturbations and keep improvements
    # Or use scipy.optimize if available
    
    # Let's use a simple coordinate descent / gradient approach
    # Actually, let's use scipy.optimize.minimize for better results
    
    from scipy.optimize import minimize
    
    def objective(c_opt):
        """Objective function for optimization."""
        if time.time() > deadline:
            return best_R  # Return best known value if deadline exceeded
        return compute_R(c_opt)
    
    # Try multiple starting points
    best_result = None
    best_obj = np.inf
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        if restart == 0:
            c_start = np.ones(21, dtype=np.float64)
        else:
            # Random initialization with some structure
            np.random.seed(42 + restart)
            c_start = np.random.randn(21) * 0.5 + np.ones(21) * 0.5
        
        try:
            result = minimize(
                objective,
                c_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception as e:
            pass
    
    if best_result is not None:
        c = best_result
    
    # Final validation
    R_final = compute_R(c)
    if not np.isfinite(R_final) or R_final > best_R:
        c = best_c
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
    
    return c


# EVOLVE-BLOCK-END
