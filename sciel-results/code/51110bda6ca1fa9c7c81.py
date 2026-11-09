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
    
    # Precompute for each theta_i:
    # base_i = N_0 * P_10^0(x_i)
    # coeff_m_i = sqrt(2) * N_m * P_10^m(x_i) for m=1..10
    
    base = N[0] * P_legendre[0]  # shape (70,)
    coeff_m = np.zeros((10, 70))  # coeff_m[m-1, i] = sqrt(2) * N_m * P_10^m(x_i)
    for m in range(1, 11):
        coeff_m[m-1] = np.sqrt(2.0) * N[m] * P_legendre[m]
    
    # Weights for quadrature: weights[i, j] = w_gl[i] * dphi
    weights = np.outer(w_gl, np.full(n_phi, dphi))  # shape (70, 130)
    
    def evaluate_f(c_vec):
        """Evaluate f on the quadrature grid."""
        c0 = c_vec[10]  # m = 0
        f = np.zeros((70, 130), dtype=np.float64)
        f += base[:, np.newaxis] * c0
        for m in range(1, 11):
            c_pos = c_vec[10 + m]  # coefficient for +m
            c_neg = c_vec[10 - m]  # coefficient for -m
            f += coeff_m[m-1, :, np.newaxis] * (c_pos * cos_m_phi[m, np.newaxis, :] + c_neg * sin_m_phi[m, np.newaxis, :])
        return f
    
    def compute_R_and_grad(c_vec):
        """Compute R = ||f||_4 / ||f||_2 and its gradient."""
        f = evaluate_f(c_vec)
        
        f2 = f ** 2
        f4 = f ** 4
        
        norm2_sq = np.sum(f2 * weights)  # ||f||_2^2
        norm4 = np.sum(f4 * weights)     # ||f||_4^4
        
        if norm2_sq <= 0 or norm4 <= 0:
            return np.inf, np.zeros(21)
            
        norm2 = np.sqrt(norm2_sq)
        norm4_val = np.sqrt(np.sqrt(norm4))  # ||f||_4
        
        R = norm4_val / norm2
        
        # Gradient computation
        # R = A^(1/4) / B^(1/2) where A = ||f||_4^4, B = ||f||_2^2
        # dR/dc_m = (1/4) A^(-3/4) B^(-1/2) dA/dc_m - (1/2) A^(1/4) B^(-3/2) dB/dc_m
        # dA/dc_m = 4 * sum(w_ij * f_ij^3 * Y_m_ij)
        # dB/dc_m = 2 * sum(w_ij * f_ij * Y_m_ij)
        
        # Precompute weighted f^3 and f
        wf3 = weights * f ** 3  # shape (70, 130)
        wf = weights * f        # shape (70, 130)
        
        grad = np.zeros(21, dtype=np.float64)
        
        # m = 0
        Y0 = base[:, np.newaxis]  # shape (70, 1)
        dA_0 = 4.0 * np.sum(wf3 * Y0)
        dB_0 = 2.0 * np.sum(wf * Y0)
        
        # m > 0 and m < 0
        for m in range(1, 11):
            # Y_m (cos term)
            Ym_cos = coeff_m[m-1, :, np.newaxis] * cos_m_phi[m, np.newaxis, :]
            # Y_-m (sin term)
            Ym_sin = coeff_m[m-1, :, np.newaxis] * sin_m_phi[m, np.newaxis, :]
            
            dA_pos = 4.0 * np.sum(wf3 * Ym_cos)
            dB_pos = 2.0 * np.sum(wf * Ym_cos)
            
            dA_neg = 4.0 * np.sum(wf3 * Ym_sin)
            dB_neg = 2.0 * np.sum(wf * Ym_sin)
            
            # Store in correct positions
            # c_vec[10 + m] is for +m, c_vec[10 - m] is for -m
            grad[10 + m] = dA_pos
            grad[10 - m] = dA_neg
            
            grad[10 + m] += 0  # placeholder, will combine below
            grad[10 - m] += 0
        
        # Actually, let me redo this more carefully
        # We need to accumulate dA/dc_m and dB/dc_m for each m
        
        dA = np.zeros(21)
        dB = np.zeros(21)
        
        # m = 0
        Y0 = base[:, np.newaxis]
        dA[10] = 4.0 * np.sum(wf3 * Y0)
        dB[10] = 2.0 * np.sum(wf * Y0)
        
        for m in range(1, 11):
            Ym_cos = coeff_m[m-1, :, np.newaxis] * cos_m_phi[m, np.newaxis, :]
            Ym_sin = coeff_m[m-1, :, np.newaxis] * sin_m_phi[m, np.newaxis, :]
            
            dA[10 + m] = 4.0 * np.sum(wf3 * Ym_cos)
            dB[10 + m] = 2.0 * np.sum(wf * Ym_cos)
            
            dA[10 - m] = 4.0 * np.sum(wf3 * Ym_sin)
            dB[10 - m] = 2.0 * np.sum(wf * Ym_sin)
        
        # Now compute gradient
        A_inv34 = norm4 ** (-0.75)
        B_inv2 = norm2_sq ** (-0.5)
        A_14 = norm4 ** 0.25
        B_inv32 = norm2_sq ** (-1.5)
        
        grad = (0.25 * A_inv34 * B_inv2) * dA - (0.5 * A_14 * B_inv32) * dB
        
        return R, grad
    
    def objective(c_opt):
        """Objective function for optimization."""
        if time.time() > deadline:
            return best_R[0]
        R, _ = compute_R_and_grad(c_opt)
        return R
    
    def objective_and_grad(c_opt):
        """Objective and gradient for optimization."""
        if time.time() > deadline:
            return best_R[0], np.zeros(21)
        R, grad = compute_R_and_grad(c_opt)
        return R, grad
    
    from scipy.optimize import minimize
    
    best_result = None
    best_obj = np.inf
    
    # Try multiple starting points
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
            
        if restart == 0:
            c_start = np.ones(21, dtype=np.float64)
        elif restart == 1:
            c_start = np.zeros(21, dtype=np.float64)
            c_start[10] = 1.0  # only m=0
        elif restart == 2:
            c_start = np.zeros(21, dtype=np.float64)
            c_start[10] = 1.0
            c_start[11] = 1.0  # m=1
        else:
            np.random.seed(42 + restart)
            c_start = np.random.randn(21) * 0.5 + np.ones(21) * 0.5
        
        try:
            result = minimize(
                objective_and_grad,
                c_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                    'maxfun': 5000
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception as e:
            pass
    
    if best_result is not None:
        c = best_result
    else:
        c = np.ones(21, dtype=np.float64)
    
    # Final validation
    R_final, _ = compute_R_and_grad(c)
    if not np.isfinite(R_final) or R_final > best_obj:
        c = best_result if best_result is not None else np.ones(21, dtype=np.float64)
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
    
    return c


# EVOLVE-BLOCK-END
