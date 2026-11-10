# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize


def _precompute_basis():
    """Precompute the quadrature grid and the 21x21 Gram matrix G where
    G_ij = sum_q w_q Y_i(x_q) Y_j(x_q) so that ||f||_2^2 = c^T G c.
    Also store the basis values at quadrature points for fast f^4 evaluation."""
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes in cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], w are weights for integral over [-1,1]
    # The surface measure: dOmega = sin(theta) dtheta dphi = -d(cos theta) dphi
    # So integral over sphere = integral_{-1}^{1} integral_0^{2pi} f(x,phi) dphi dx
    # With Gauss-Legendre: sum_i w_i f(x_i, phi_j) * (2pi/n_phi)
    
    # Phi values
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    phi_weight = 2 * np.pi / n_phi

    # Build theta, phi grid
    # x = cos(theta), so theta = arccos(x)
    # We need P_10^m(cos(theta)) = P_10^m(x)
    
    # cos and sin of m*phi for m = 0..10
    # For m=0: cos(0*phi)=1, sin(0*phi)=0
    # For m>0: need cos(m*phi_j) and sin(m*phi_j)
    
    # Basis functions:
    # Y_0 = N_0 P_10^0(x)
    # Y_m = sqrt(2) N_m P_10^m(x) cos(m*phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(x) sin(|m|*phi) for m < 0
    
    # Coefficients c_m ordered m = -10..10, index i = m + 10
    # So index 0 -> m=-10, index 10 -> m=0, index 20 -> m=10
    
    # Precompute N_m for m = 0..10
    N = np.zeros(k + 1)
    for m in range(k + 1):
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * 
                       np.math.factorial(k - m) / np.math.factorial(k + m))
    
    # Precompute P_10^m(x) for m = 0..10 at all x_i
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        # lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase
        P[:, m] = lpmv(m, k, x)
    
    # Build basis matrix B of shape (n_theta * n_phi, 21)
    # B[q, i] = Y_i(x_{q//n_phi}, phi_{q%n_phi})
    npts = n_theta * n_phi
    B = np.zeros((npts, 2 * k + 1))
    
    # Weights for each point
    # weight_q = w_{q//n_phi} * phi_weight
    w_theta = w  # shape (n_theta,)
    weights = np.repeat(w_theta * phi_weight, n_phi)  # shape (npts,)
    
    for i in range(2 * k + 1):
        m = i - k
        abs_m = abs(m)
        for it in range(n_theta):
            for ip in range(n_phi):
                q = it * n_phi + ip
                val = P[it, abs_m]
                if m == 0:
                    B[q, i] = N[0] * val
                elif m > 0:
                    B[q, i] = np.sqrt(2) * N[abs_m] * val * np.cos(m * phi[ip])
                else:
                    B[q, i] = np.sqrt(2) * N[abs_m] * val * np.sin(abs_m * phi[ip])
    
    # Gram matrix G: G[i,j] = sum_q weights[q] * B[q,i] * B[q,j]
    # G = B^T W B where W is diagonal weights
    G = (B * weights[:, None]).T @ B  # shape (21, 21)
    
    return B, weights, G, N, P, x, phi, n_theta, n_phi


def _objective_and_gradient(c, B, weights, G):
    """Compute R = ||f||_4 / ||f||_2 and its gradient.
    
    ||f||_2^2 = c^T G c
    ||f||_4^4 = sum_q weights[q] * (B c)_q^4
    
    R = (sum_q w_q f_q^4)^(1/4) / (c^T G c)^(1/2)
    
    To minimize R, we can minimize R^4 = (sum_q w_q f_q^4) / (c^T G c)^2
    or minimize log(R) = (1/4) log(sum w f^4) - (1/2) log(c^T G c)
    
    Let's minimize F = (sum_q w_q f_q^4) / (c^T G c)^2 which is R^4.
    """
    f = B @ c  # shape (npts,)
    f2 = f * f
    f4 = f2 * f2
    norm4_4 = np.sum(weights * f4)  # ||f||_4^4
    norm2_2 = c @ G @ c  # ||f||_2^2
    
    if norm2_2 <= 0:
        return 1e10, np.zeros_like(c)
    
    # F = norm4_4 / norm2_2^2
    F = norm4_4 / (norm2_2 * norm2_2)
    
    # Gradient of F with respect to c:
    # F = A / B^2 where A = sum w f^4, B = c^T G c
    # dA/dc_i = sum_q w_q * 4 f_q^3 * B_q,i
    # dB/dc_i = 2 (G c)_i
    # dF/dc_i = (dA/dc_i * B^2 - A * 2 B * dB/dc_i) / B^4
    #         = (dA/dc_i * B - A * 2 * dB/dc_i) / B^3
    #         = (dA/dc_i / B - 2 A dB/dc_i / B^2) / B
    # Let's be more careful:
    # F = A * B^{-2}
    # dF/dc_i = dA/dc_i * B^{-2} + A * (-2) B^{-3} * dB/dc_i
    #         = dA/dc_i / B^2 - 2 A dB/dc_i / B^3
    
    dA_dc = 4 * np.sum(weights * f4 / f2 * f * B, axis=0) if np.any(f2 > 1e-30) else np.zeros(21)
    # Actually: d/dc_i sum_q w_q f_q^4 = sum_q w_q * 4 f_q^3 * (B c)_q,i = 4 sum_q w_q f_q^3 B_q,i
    # But f_q^3 / f_q = f_q^2, so 4 sum_q w_q f_q^3 B_q,i = 4 sum_q w_q f_q^2 * f_q * B_q,i
    # Let's just compute directly:
    f3 = f2 * f
    dA_dc = 4 * (B.T @ (weights * f3))  # shape (21,)
    
    Gc = G @ c  # shape (21,)
    dB_dc = 2 * Gc  # shape (21,)
    
    B2 = norm2_2
    B3 = B2 * B2
    
    grad_F = dA_dc / (B2 * B2) - 2 * norm4_4 * dB_dc / (B2 * B2 * B2)
    
    return F, grad_F


def rudin_problem10() -> np.ndarray:
    k = 10
    deadline = time.time() + 580  # safety margin
    
    B, weights, G, N, P, x, phi, n_theta, n_phi = _precompute_basis()
    
    # Normalize G to have unit diagonal for better conditioning
    # Actually, let's just use it as is
    
    # Initial guess: all ones
    c0 = np.ones(2 * k + 1, dtype=np.float64)
    
    best_c = c0.copy()
    best_F = _objective_and_gradient(c0, B, weights, G)[0]
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    
    starts = [c0.copy()]
    # Add some structured starts
    for trial in range(20):
        c_trial = rng.randn(2 * k + 1)
        c_trial /= np.linalg.norm(c_trial) * np.sqrt(21)  # normalize
        starts.append(c_trial)
    
    # Also try alternating sign patterns
    for m in range(1, 11):
        c_alt = np.ones(2 * k + 1)
        c_alt[m + k] = -1  # flip one coefficient
        starts.append(c_alt.copy())
        c_alt2 = np.ones(2 * k + 1)
        c_alt2[-(m) + k] = -1  # flip negative m
        starts.append(c_alt2.copy())
    
    # Run optimization from each start
    for c_init in starts:
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                lambda c: _objective_and_gradient(c, B, weights, G),
                c_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_F:
                F_val, _ = _objective_and_gradient(res.x, B, weights, G)
                if F_val < best_F:
                    best_F = F_val
                    best_c = res.x.copy()
        except Exception:
            continue
    
    # Polish with tighter tolerance
    if time.time() < deadline - 30:
        try:
            res = minimize(
                lambda c: _objective_and_gradient(c, B, weights, G),
                best_c,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_F:
                F_val, _ = _objective_and_gradient(res.x, B, weights, G)
                if F_val < best_F:
                    best_F = F_val
                    best_c = res.x.copy()
        except Exception:
            pass
    
    # Scale coefficients so max|c| is in a reasonable range
    max_abs = np.max(np.abs(best_c))
    if max_abs > 1e5:
        best_c = best_c / max_abs * 1e3
    elif max_abs < 1e-3:
        best_c = best_c / max_abs * 1e0
    
    # Final validation
    assert len(best_c) == 21
    assert not np.any(np.isnan(best_c))
    assert not np.any(np.isinf(best_c))
    assert np.max(np.abs(best_c)) >= 1e-3
    assert np.max(np.abs(best_c)) <= 1e5
    
    return best_c


# EVOLVE-BLOCK-END
