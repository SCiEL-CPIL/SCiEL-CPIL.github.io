# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
import time


def _build_harmonics_grid(n_theta=70, n_phi=130):
    """
    Build the real orthonormal degree-10 spherical harmonics evaluated on
    a 70-point Gauss-Legendre (in cos theta) x 130-point (in phi) grid.
    
    Returns:
        H: array of shape (21, n_theta, n_phi) where H[m_idx] is Y_{m}
        weights: array of shape (n_theta, n_phi) with the quadrature weights
        (such that sum(f * weights) approximates integral over sphere)
    """
    k = 10
    
    # Gauss-Legendre nodes and weights in [-1, 1] for cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is cos(theta), w are the weights for integral over [-1,1]
    
    # Phi nodes: equally spaced
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Weights on the sphere: dOmega = dphi * d(cos theta)
    # Integral over sphere = sum_theta w_theta * sum_phi (dphi) * f
    dphi = 2 * np.pi / n_phi
    # weights[i,j] = w[i] * dphi
    weights = np.outer(w, np.ones(n_phi)) * dphi
    
    # Build the harmonics
    # m goes from -10 to 10, index 0 corresponds to m=-10, index 20 to m=10
    H = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    for idx in range(21):
        m = idx - 10  # m from -10 to 10
        
        if m == 0:
            # Y_0 = N_0 * P_10^0(cos theta)
            N_0 = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10) / factorial(10))
            P = lpmv(0, 10, x)  # P_10^0(cos theta) with Condon-Shortley
            H[0, :, :] = N_0 * P[:, np.newaxis]
        
        elif m > 0:
            # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m * phi)
            N_m = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - m) / factorial(10 + m))
            P = lpmv(m, 10, x)  # P_10^m(cos theta) with Condon-Shortley
            cos_part = np.cos(m * phi)
            H[idx, :, :] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_part[np.newaxis, :]
        
        else:
            # Y_m = sqrt(2) * N_{|m|} * P_10^{|m|}(cos theta) * sin(|m| * phi)
            abs_m = -m
            N_m = np.sqrt(21.0 / (4.0 * np.pi) * factorial(10 - abs_m) / factorial(10 + abs_m))
            P = lpmv(abs_m, 10, x)
            sin_part = np.sin(abs_m * phi)
            H[idx, :, :] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_part[np.newaxis, :]
    
    return H, weights


def _compute_ratio(coeffs, H, weights):
    """
    Compute R = ||f||_4 / ||f||_2 for f = sum_m c_m Y_m.
    
    coeffs: shape (21,)
    H: shape (21, n_theta, n_phi)
    weights: shape (n_theta, n_phi)
    """
    # f = sum_m c_m * H[m]
    # coeffs has shape (21,), H has shape (21, nt, np)
    f = np.einsum('m,mij->ij', coeffs, H)
    
    # ||f||_2^2 = sum f^2 * weights
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    # ||f||_4^4 = sum |f|^4 * weights
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _objective_and_grad(coeffs, H, weights):
    """
    Objective: minimize R = ||f||_4 / ||f||_2
    Also compute gradient with respect to coeffs.
    
    Let f = H^T c (vectorized)
    ||f||_2^2 = c^T (H^T W H) c = c^T G c
    ||f||_4^4 = sum (f_i^4) w_i
    
    R = (sum f_i^4 w_i)^{1/4} / (c^T G c)^{1/2}
    
    We minimize R. Let's minimize R^4 to avoid roots:
    J = R^4 = (sum f_i^4 w_i) / (c^T G c)^2
    
    Actually, let's just minimize R directly.
    
    f = H c (where H is (nt*np, 21) matrix)
    Let A = H reshaped to (N, 21) where N = nt*np
    f = A c
    ||f||_2^2 = f^T W f = c^T (A^T W A) c
    ||f||_4^4 = sum_i (f_i^4) w_i
    
    R = (sum f_i^4 w_i)^{1/4} / (c^T A^T W A c)^{1/2}
    
    Let S = sum f_i^4 w_i
    Let Q = c^T G c where G = A^T W A
    
    R = S^{1/4} / Q^{1/2}
    
    dR/dc = (1/4) S^{-3/4} * dS/dc / Q^{1/2} - S^{1/4} * (1/2) Q^{-3/2} * dQ/dc
    
    dS/dc = sum_i 4 f_i^3 w_i * (d f_i / d c)
    f_i = sum_j A_{i,j} c_j
    d f_i / d c = A_{i,:}
    dS/dc = 4 * sum_i f_i^3 w_i A_{i,:} = 4 * A^T (w * f^3)
    
    dQ/dc = 2 G c
    
    So dR/dc = (1/4) S^{-3/4} Q^{-1/2} * 4 A^T(w f^3) - S^{1/4} (1/2) Q^{-3/2} * 2 G c
             = S^{-3/4} Q^{-1/2} A^T(w f^3) - S^{1/4} Q^{-3/2} G c
             = (1/(S^{3/4} Q^{1/2})) [A^T(w f^3) - (S^{1/4} Q^{-1}) (S^{3/4} Q^{1/2}) G c]
    
    Wait, let me redo this more carefully.
    
    R = S^{1/4} Q^{-1/2}
    
    dR/dc = (1/4) S^{-3/4} Q^{-1/2} dS/dc + S^{1/4} (-1/2) Q^{-3/2} dQ/dc
          = (1/4) S^{-3/4} Q^{-1/2} * 4 A^T(w f^3) - (1/2) S^{1/4} Q^{-3/2} * 2 G c
          = S^{-3/4} Q^{-1/2} A^T(w f^3) - S^{1/4} Q^{-3/2} G c
    
    Let me factor out S^{-3/4} Q^{-3/2}:
    = S^{-3/4} Q^{-3/2} [Q A^T(w f^3) - S^{1/4} S^{3/4} G c]
    = S^{-3/4} Q^{-3/2} [Q A^T(w f^3) - S G c]
    
    Hmm, that's not quite right either. Let me just compute it directly.
    """
    N, nt, nph = H.shape
    A = H.reshape(N, nt * nph).T  # (nt*nph, 21)
    W = weights.reshape(nt * nph)
    
    f = A.T @ coeffs  # (nt*nph,)
    f2 = f * f
    f3 = f2 * f
    
    S = np.sum(f4 := f2 * f2 * W)  # sum f^4 w
    G = A.T @ (W * A)  # (21, 21) - precomputed? No, let's compute Q directly
    Q = coeffs @ (A.T @ (W * (A @ coeffs)))
    
    # Actually Q = c^T A^T W A c = f^T W f
    Q = np.sum(f2 * W)
    
    if S <= 0 or Q <= 0:
        return np.inf, np.zeros(21)
    
    # dS/dc = 4 A^T (w * f^3)
    dS_dc = 4.0 * (A.T @ (W * f3))
    
    # dQ/dc = 2 A^T (W * f)
    dQ_dc = 2.0 * (A.T @ (W * f))
    
    S_q34 = S ** 0.75
    Q_q12 = np.sqrt(Q)
    Q_q32 = Q ** 1.5
    
    grad = (dS_dc / (4.0 * S_q34 * Q_q12)) - (S ** 0.25 * dQ_dc / (2.0 * Q_q32))
    
    R = (S ** 0.25) / Q_q12
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a degree-10 spherical harmonic
    that minimize R = ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds
    
    # Build the harmonics grid
    H, weights = _build_harmonics_grid(n_theta=70, n_phi=130)
    
    # Precompute A and W for gradient computations
    nt, nph = H.shape[1], H.shape[2]
    A = H.reshape(21, nt * nph).T  # (nt*nph, 21)
    W = weights.reshape(nt * nph)
    
    def obj_grad(coeffs):
        """Objective and gradient for scipy.optimize.minimize."""
        f = A @ coeffs
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        
        S = np.sum(f4 * W)
        Q = np.sum(f2 * W)
        
        if S <= 0 or Q <= 0:
            return np.inf, np.zeros(21)
        
        # R = S^{1/4} / Q^{1/2}
        R = (S ** 0.25) / np.sqrt(Q)
        
        # Gradient
        dS_dc = 4.0 * (A.T @ (W * f3))
        dQ_dc = 2.0 * (A.T @ (W * f))
        
        S_q34 = S ** 0.75
        Q_q12 = np.sqrt(Q)
        Q_q32 = Q ** 1.5
        
        grad = (dS_dc / (4.0 * S_q34 * Q_q12)) - ((S ** 0.25) * dQ_dc / (2.0 * Q_q32))
        
        return R, grad
    
    # Initial guess: all ones
    best_coeffs = np.ones(21, dtype=np.float64)
    best_R, _ = obj_grad(best_coeffs)
    
    # Try multiple random initializations and optimize
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        if restart == 0:
            x0 = np.ones(21, dtype=np.float64)
        else:
            # Random initialization with some structure
            x0 = np.random.randn(21)
            # Add some symmetry-breaking
            x0[10] = np.random.randn()  # m=0 term
        
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_R:
                if np.isfinite(result.fun) and result.fun < best_R:
                    best_R = result.fun
                    best_coeffs = result.x.copy()
                    
        except Exception as e:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - safety_margin:
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                obj_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-15}
            )
            
            if np.isfinite(result.fun) and result.fun < best_R:
                best_R = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
    
    # Validate the result
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs


# EVOLVE-BLOCK-END
