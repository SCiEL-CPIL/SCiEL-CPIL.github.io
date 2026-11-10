# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time


def _build_quadrature_and_harmonics():
    """Build Gauss-Legendre quadrature grid and precompute harmonic values."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x is in [-1, 1], this is cos(theta)
    # theta = arccos(x), so dtheta = -dx / sqrt(1-x^2)
    # The surface measure is sin(theta) dtheta dphi = -dx dphi
    # So we integrate over x from -1 to 1 with weight 1 (the sin(theta)dtheta = -dx)
    # Total integral over sphere: ∫_{-1}^{1} ∫_0^{2π} f dx dphi = 2π * ∫_{-1}^{1} f dx
    # But we need to be careful: the standard surface measure dΩ = sin(θ)dθ dφ
    # With x = cos(θ), dΩ = -dx dφ, so ∫∫ f dΩ = ∫_{-1}^{1} ∫_0^{2π} f(x, φ) dφ dx
    
    # So the quadrature is: sum_i w_i * sum_j f(x_i, phi_j) * (2π/n_phi)
    # where w_i are the Gauss-Legendre weights
    
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    phi_dphi = 2 * np.pi / n_phi
    
    # Build 2D grid
    x_grid, phi_grid = np.meshgrid(x, phi, indexing='ij')
    
    # Precompute harmonic values for each m
    # Y_0 = N_0 * P_10^0(x)
    # Y_m = sqrt(2) * N_m * P_10^m(x) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^{|m|}(x) * sin(|m|*phi) for m < 0
    
    N = np.zeros(2 * k + 1)
    for m in range(k + 1):
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Use log-gamma for stability
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + 
                       (scipy_scipy_log_factorial(10 - m) - scipy_scipy_log_factorial(10 + m)))
        N[m] = np.exp(log_N)
    
    # Store harmonic values in a 3D array: (n_theta, n_phi, 21)
    # Index: i-th coefficient corresponds to m = i - 10
    H = np.zeros((n_theta, n_phi, 2 * k + 1))
    
    for m_abs in range(k + 1):
        # Compute P_10^m_abs(x) with Condon-Shortley phase
        P = lpmv(m_abs, 10, x)  # This includes Condon-Shortley phase
        P_grid = np.tile(P[:, np.newaxis], (1, n_phi))
        
        if m_abs == 0:
            # Y_0
            H[:, :, 10] = N[0] * P_grid
        else:
            # Y_m for m > 0: sqrt(2) * N_m * P * cos(m*phi)
            cos_part = np.cos(m_abs * phi_grid)
            H[:, :, 10 + m_abs] = np.sqrt(2) * N[m_abs] * P_grid * cos_part
            
            # Y_m for m < 0: sqrt(2) * N_|m| * P * sin(|m|*phi)
            sin_part = np.sin(m_abs * phi_grid)
            H[:, :, 10 - m_abs] = np.sqrt(2) * N[m_abs] * P_grid * sin_part
    
    return x, w, phi, phi_dphi, H


def scipy_scipy_log_factorial(n):
    """Compute log(n!) using scipy or fallback."""
    from scipy.special import gammaln
    return gammaln(n + 1)


def _compute_ratio(coeffs, H, w, phi_dphi):
    """Compute L4/L2 ratio given coefficients."""
    # f = sum_m c_m * Y_m
    # H shape: (n_theta, n_phi, 21)
    # coeffs shape: (21,)
    f = np.tensordot(coeffs, H, axes=(0, 2))  # shape (n_theta, n_phi)
    
    # Norms: ∫∫ |f|^p dΩ = ∫_{-1}^{1} ∫_0^{2π} |f|^p dφ dx
    # Quadrature: sum_i w_i * sum_j |f_ij|^p * phi_dphi
    
    f2 = f ** 2
    f4 = f2 ** 2
    
    L2_sq = np.sum(w[:, np.newaxis] * f2 * phi_dphi)
    L4 = np.sum(w[:, np.newaxis] * f4 * phi_dphi)
    
    if L2_sq <= 0:
        return np.inf, np.zeros(21)
    
    L2 = np.sqrt(L2_sq)
    L4_norm = L4 ** 0.25
    
    ratio = L4_norm / L2
    
    # Gradient of ratio with respect to coeffs
    # d(L4/L2)/dc = (1/L2) * dL4/dc - (L4/L2^2) * dL2/dc
    # dL4/dc = 4 * ∫∫ |f|^2 * f * Y_m dΩ * 0.25 * 4 = ∫∫ f^3 * Y_m dΩ * 4
    # Actually: L4 = (∫|f|^4)^(1/4), dL4/dc = (1/4)(∫|f|^4)^(-3/4) * 4∫|f|^2*f*Y_m = ∫f^3*Y_m / L4^3
    # L2 = (∫|f|^2)^(1/2), dL2/dc = (1/2)(∫|f|^2)^(-1/2) * 2∫f*Y_m = ∫f*Y_m / L2
    
    # d(ratio)/dc = (1/L2) * dL4/dc - (L4/L2^2) * dL2/dc
    # = (1/L2) * (∫f^3*Y_m / L4^3) - (L4/L2^2) * (∫f*Y_m / L2)
    # = ∫f^3*Y_m / (L2 * L4^3) - L4 * ∫f*Y_m / L2^3
    
    f3 = f2 * f
    f1 = f
    
    # ∫f^3 * Y_m dΩ for each m
    grad_L4 = np.tensordot(w[:, np.newaxis] * f3 * phi_dphi, H, axes=([0, 1], [0, 1]))
    grad_L2 = np.tensordot(w[:, np.newaxis] * f1 * phi_dphi, H, axes=([0, 1], [0, 1]))
    
    grad_ratio = grad_L4 / (L2 * L4**3) - L4 * grad_L2 / L2**3
    
    return ratio, grad_ratio


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 that minimize L4/L2 ratio
    of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Build quadrature and harmonics
    x, w, phi, phi_dphi, H = _build_quadrature_and_harmonics()
    
    # Initial guess: all ones
    coeffs0 = np.ones(21)
    
    # Compute initial ratio
    ratio0, grad0 = _compute_ratio(coeffs0, H, w, phi_dphi)
    
    best_coeffs = coeffs0.copy()
    best_ratio = ratio0
    
    # Use L-BFGS-B optimization
    def objective_and_grad(c):
        r, g = _compute_ratio(c, H, w, phi_dphi)
        return r, g
    
    # Try multiple initializations
    initializations = [
        np.ones(21),
        np.zeros(21),
    ]
    initializations[1][10] = 1.0  # Y_0 only
    
    # Add random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        init = rng.standard_normal(21)
        initializations.append(init)
    
    for init in initializations:
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success and result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Additional refinement with more iterations if time permits
    if time.time() < deadline - 30 and best_ratio < np.inf:
        try:
            result = minimize(
                objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.success and result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.ones(21)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e3
    
    return best_coeffs


# EVOLVE-BLOCK-END
