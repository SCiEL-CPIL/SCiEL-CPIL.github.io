# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
import numpy.polynomial.legendre as npl


def _build_basis():
    """Build real orthonormal degree-10 spherical harmonics on a quadrature grid."""
    k = 10
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    mu, w = npl.leggauss(n_theta)  # mu in [-1, 1], w are weights
    # theta = arccos(mu), dOmega = dmu dphi (since sin(theta)dtheta dphi = -dmu dphi)
    # Actually: dOmega = sin(theta) dtheta dphi, and mu = cos(theta), so sin(theta)dtheta = -dmu
    # The integral over sphere: ∫∫ f dOmega = ∫_{-1}^{1} ∫_0^{2π} f(mu, phi) dphi dmu
    # (the sin(theta) is absorbed into the mu integration)
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Create grid
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    
    # Build basis functions Y_m for m = -10..10
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    basis = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    for m_idx in range(21):
        m = m_idx - 10  # m ranges from -10 to 10
        
        # Compute N_m
        abs_m = abs(m)
        # (10-abs_m)! / (10+abs_m)!
        num = 1.0
        for i in range(10 - abs_m + 1, 11):
            num *= i
        den = 1.0
        for i in range(10 - abs_m + 1, 10 + abs_m + 1):
            den *= i
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * num / den)
        
        # Compute associated Legendre P_10^|m|(mu) with Condon-Shortley phase
        P = lpmv(abs_m, 10, mu)  # This includes Condon-Shortley phase (-1)^m
        
        if m == 0:
            basis[m_idx] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return basis, mu, w, phi, dphi


def _compute_R(c, basis, mu, w, phi, dphi):
    """Compute R = ||f||_4 / ||f||_2 using quadrature."""
    # f = sum c_m Y_m
    f = np.tensordot(c, basis, axes=(0, 0))  # shape (n_theta, n_phi)
    
    # Quadrature weights: dmu * dphi
    # w has shape (n_theta,), dphi is scalar
    weights = np.outer(w, np.full(len(phi), dphi))
    
    # ||f||_2^2 = ∫∫ f^2 dOmega
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    # ||f||_4^4 = ∫∫ f^4 dOmega
    f4 = f * f * f * f
    norm4_4 = np.sum(f4 * weights)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _compute_grad(c, basis, mu, w, phi, dphi):
    """Compute gradient of R with respect to c."""
    f = np.tensordot(c, basis, axes=(0, 0))
    
    weights = np.outer(w, np.full(len(phi), dphi))
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    f4 = f * f * f * f
    norm4_4 = np.sum(f4 * weights)
    
    if norm2_sq <= 0:
        return np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    R = norm4 / norm2
    
    # dR/dc_j = (1/norm2) * d(norm4)/dc_j - (norm4/norm2^2) * d(norm2)/dc_j
    # d(norm4)/dc_j = (1/4) * norm4_4^(-3/4) * d(norm4_4)/dc_j
    # d(norm4_4)/dc_j = 4 * ∫∫ f^3 Y_j dOmega
    # d(norm2)/dc_j = (1/norm2) * (1/2) * d(norm2_sq)/dc_j
    # d(norm2_sq)/dc_j = 2 * ∫∫ f Y_j dOmega
    
    # Precompute integrals
    f3 = f * f * f
    int_f3_Y = np.array([np.sum(f3 * basis[j] * weights) for j in range(21)])
    int_f_Y = np.array([np.sum(f * basis[j] * weights) for j in range(21)])
    
    d_norm4_4_dc = 4.0 * int_f3_Y
    d_norm4_dc = 0.25 * norm4_4 ** (-0.75) * d_norm4_4_dc
    
    d_norm2_sq_dc = 2.0 * int_f_Y
    d_norm2_dc = d_norm2_sq_dc / (2.0 * norm2)
    
    grad = d_norm4_dc / norm2 - (norm4 / norm2 ** 2) * d_norm2_dc
    
    return grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    # Build basis
    basis, mu, w, phi, dphi = _build_basis()
    
    # Initial guess: all ones
    c = np.ones(21, dtype=np.float64)
    best_R = _compute_R(c, basis, mu, w, phi, dphi)
    best_c = c.copy()
    
    # Gradient descent with adaptive step size
    lr = 0.1
    momentum = np.zeros(21, dtype=np.float64)
    beta = 0.9
    
    step = 0
    while time.time() < deadline:
        R = _compute_R(c, basis, mu, w, phi, dphi)
        if R < best_R:
            best_R = R
            best_c = c.copy()
        
        grad = _compute_grad(c, basis, mu, w, phi, dphi)
        
        # Momentum update
        momentum = beta * momentum - lr * grad
        c_new = c + momentum
        
        # Check if valid
        if not np.all(np.isfinite(c_new)):
            lr *= 0.5
            momentum *= 0.5
            continue
        
        R_new = _compute_R(c_new, basis, mu, w, phi, dphi)
        
        if R_new < R:
            c = c_new
            # Try increasing learning rate
            if step % 100 == 0:
                lr = min(lr * 1.05, 1.0)
        else:
            # Reduce learning rate and revert
            lr *= 0.9
            momentum *= 0.5
            c_new = c  # keep old c
        
        step += 1
        
        # Periodically check time
        if step % 50 == 0 and time.time() > deadline - 30:
            break
    
    # Final check
    R_final = _compute_R(best_c, basis, mu, w, phi, dphi)
    
    # Validate
    if best_c.shape != (21,) or not np.all(np.isfinite(best_c)):
        return np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale
        if max_abs > 0:
            best_c = best_c * (1.0 / max_abs)
        else:
            return np.ones(21, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
