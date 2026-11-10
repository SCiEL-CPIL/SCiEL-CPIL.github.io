# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import time


def _compute_quadrature():
    """Compute 70x130 quadrature points on the sphere."""
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl in (-1, 1), w_gl are weights
    # Map to cos(theta): cos_theta = x_gl
    cos_theta = x_gl
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    
    # Surface element: dOmega = sin(theta) dtheta dphi
    # In terms of cos_theta: dOmega = d(cos_theta) dphi (with sign, but we use abs)
    # Actually: dtheta = -d(cos_theta)/sin(theta), so sin(theta)*dtheta*dphi = d(cos_theta)*dphi
    # So the weight is just w_gl * dphi (the sin(theta) cancels with the Jacobian)
    
    # Create meshgrid
    CT, P = np.meshgrid(cos_theta, phi, indexing='ij')
    ST = np.broadcast_to(sin_theta[:, None], (70, 130))
    
    # Weights: w_gl[i] * dphi for each (i, j)
    W = np.broadcast_to(w_gl[:, None], (70, 130)) * dphi
    
    return CT, ST, P, W, sin_theta, cos_theta


def _compute_basis(cos_theta, sin_theta, phi):
    """
    Compute real orthonormal degree-10 spherical harmonics.
    Returns array of shape (21, 70, 130) where index i corresponds to m = i - 10.
    
    Y_0 = N_0 * P_10^0(cos theta)
    Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    
    N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    """
    k = 10
    n_basis = 2 * k + 1
    shape = cos_theta.shape
    basis = np.zeros((n_basis,) + shape, dtype=np.float64)
    
    # Precompute factorials
    from math import factorial, lgamma
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # Use log for stability
    for m in range(0, k + 1):
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(lgamma(k - m + 1) - lgamma(k + m + 1)))
        idx = m  # m = 0 -> index 0, m=1 -> index 1, ..., m=10 -> index 10
        # But our ordering is m = -10..10, so index i corresponds to m = i - 10
        # m=0 -> index 10, m=1 -> index 11, ..., m=10 -> index 20
        # m=-1 -> index 9, m=-10 -> index 0
        
        if m == 0:
            # Y_0 = N_0 * P_10^0(cos theta)
            P = lpmv(0, k, cos_theta)  # Condon-Shortley phase included
            basis[idx + 10] = N_m * P
        else:
            P = lpmv(m, k, cos_theta)  # Condon-Shortley phase included
            basis[m + 10] = np.sqrt(2.0) * N_m * P * np.cos(m * phi)
            basis[-m + 10] = np.sqrt(2.0) * N_m * P * np.sin(m * phi)
    
    return basis


def _compute_ratio(c, basis_flat, W_flat):
    """
    Compute R = ||f||_4 / ||f||_2.
    c: (21,) coefficient vector
    basis_flat: (21, N) flattened basis
    W_flat: (N,) weights
    """
    f = basis_flat @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(W_flat * f2)
    norm4_4 = np.sum(W_flat * f4)
    
    if norm2_sq < 1e-30:
        return 1e10
    
    R = (norm4_4 / norm2_sq) ** 0.5 / (norm2_sq / (4.0 * np.pi)) ** 0.25
    # Wait, let me reconsider. ||f||_2 = sqrt(int f^2 dOmega), ||f||_4 = (int f^4 dOmega)^(1/4)
    # R = ||f||_4 / ||f||_2
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = (norm4_4) ** 0.25
    R = norm4 / norm2
    return R


def _compute_ratio_and_grad(c, basis_flat, W_flat):
    """
    Compute R and its gradient w.r.t. c.
    R = ||f||_4 / ||f||_2 = (int f^4)^(1/4) / (int f^2)^(1/2)
    
    Let A = int f^4 dOmega, B = int f^2 dOmega
    R = A^(1/4) / B^(1/2)
    
    dR/dc = (1/4) A^(-3/4) * (dA/dc) / B^(1/2) - A^(1/4) * (1/2) B^(-3/2) * (dB/dc)
    
    dA/dc = 4 * int f^3 (df/dc) dOmega = 4 * sum_i W_i f_i^3 * basis_i
    dB/dc = 2 * int f (df/dc) dOmega = 2 * sum_i W_i f_i * basis_i
    """
    f = basis_flat @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    
    A = np.sum(W_flat * f4)  # int f^4
    B = np.sum(W_flat * f2)  # int f^2
    
    if B < 1e-30:
        return 1e10, np.zeros_like(c)
    
    norm4 = A ** 0.25
    norm2 = np.sqrt(B)
    R = norm4 / norm2
    
    # dA/dc = 4 * sum_i W_i f_i^3 * basis_i[m] for each m
    # = 4 * basis_flat.T @ (W_flat * f^3)
    dA_dc = 4.0 * (basis_flat.T @ (W_flat * f**3))
    
    # dB/dc = 2 * sum_i W_i f_i * basis_i[m]
    dB_dc = 2.0 * (basis_flat.T @ (W_flat * f))
    
    # dR/dc = (1/4) A^(-3/4) B^(-1/2) dA_dc - A^(1/4) (1/2) B^(-3/2) dB_dc
    # = A^(-3/4) B^(-1/2) [ (1/4) dA_dc - (1/2) A^(1/2) B^(-1/2) dB_dc ]
    # = (1/(4 * A^(3/4) * B^(1/2))) * [ dA_dc - 2 * (A^(1/2)/B^(1/2)) * dB_dc ]
    
    # Simplify: R = A^(1/4) B^(-1/2)
    # dR/dc = (1/4) A^(-3/4) B^(-1/2) dA_dc - (1/2) A^(1/4) B^(-3/2) dB_dc
    # = (1/(4 A^(3/4) B^(1/2))) [ dA_dc - 2 A^(1/2) B^(-1/2) dB_dc ]
    # = (1/(4 A^(3/4) B^(1/2))) [ dA_dc - 2 (A/B)^(1/2) dB_dc ]
    
    factor = 1.0 / (4.0 * A**0.75 * B**0.5)
    ratio_AB = np.sqrt(A / B)
    grad = factor * (dA_dc - 2.0 * ratio_AB * dB_dc)
    
    return R, grad


def _objective(c, basis_flat, W_flat):
    """Objective: minimize R = ||f||_4 / ||f||_2"""
    R, _ = _compute_ratio_and_grad(c, basis_flat, W_flat)
    return R


def _objective_grad(c, basis_flat, W_flat):
    R, grad = _compute_ratio_and_grad(c, basis_flat, W_flat)
    return R, grad


def _norm_constraint(c):
    """Constraint: ||c||^2 = 1 (i.e., ||f||_2^2 = 1 since basis is orthonormal)"""
    return np.sum(c * c) - 1.0


def _norm_constraint_grad(c):
    return 2.0 * c


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing R = ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Setup quadrature and basis
    CT, ST, P, W, sin_theta, cos_theta = _compute_quadrature()
    basis = _compute_basis(cos_theta, sin_theta, P)
    # Flatten for matrix operations
    N = basis.shape[1] * basis.shape[2]
    basis_flat = basis.reshape(21, N)
    W_flat = W.reshape(N)
    
    # Initial best
    best_c = np.ones(21, dtype=np.float64)
    best_c /= np.linalg.norm(best_c)
    best_R = _compute_ratio(best_c, basis_flat, W_flat)
    
    # Multiple restarts with different initial conditions
    rng = np.random.default_rng(42)
    
    # Try several starting points
    starts = [best_c.copy()]
    
    # Random starts
    for _ in range(15):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
    
    # Some structured starts
    # Only m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    starts.append(c0)
    
    # Only m=10 (cos)
    c10 = np.zeros(21)
    c10[20] = 1.0
    starts.append(c10)
    
    # Only m=10 (sin)
    c10s = np.zeros(21)
    c10s[0] = 1.0
    starts.append(c10s)
    
    # m=0 and m=10
    cmix = np.zeros(21)
    cmix[10] = 1.0
    cmix[20] = 1.0
    cmix /= np.linalg.norm(cmix)
    starts.append(cmix)
    
    # Alternating signs
    calt = np.ones(21)
    calt[::2] = -1.0
    calt /= np.linalg.norm(calt)
    starts.append(calt)
    
    # Symmetric patterns
    for pattern in [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]:
        cp = np.array(pattern, dtype=np.float64)
        nrm = np.linalg.norm(cp)
        if nrm > 1e-10:
            cp /= nrm
            starts.append(cp)
    
    constraints = [
        {'type': 'eq', 'fun': _norm_constraint, 'jac': _norm_constraint_grad}
    ]
    
    # Optimize from each starting point
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 60.0:
            break
        
        try:
            result = minimize(
                _objective_grad,
                c0,
                args=(basis_flat, W_flat),
                method='L-BFGS-B',
                constraints=constraints,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            
            if result.success or result.fun < best_R:
                c_new = result.x
                # Normalize
                nrm = np.linalg.norm(c_new)
                if nrm > 1e-10:
                    c_new /= nrm
                R_new = _compute_ratio(c_new, basis_flat, W_flat)
                if R_new < best_R:
                    best_R = R_new
                    best_c = c_new.copy()
        except Exception:
            continue
    
    # Additional local refinement with tighter tolerances
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                _objective_grad,
                best_c,
                args=(basis_flat, W_flat),
                method='SLSQP',
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-14,
                    'disp': False
                }
            )
            if result.fun < best_R:
                c_new = result.x
                nrm = np.linalg.norm(c_new)
                if nrm > 1e-10:
                    c_new /= nrm
                R_new = _compute_ratio(c_new, basis_flat, W_flat)
                if R_new < best_R:
                    best_R = R_new
                    best_c = c_new.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_grad,
                best_c,
                args=(basis_flat, W_flat),
                method='SLSQP',
                constraints=constraints,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-16,
                    'disp': False
                }
            )
            if result.fun < best_R:
                c_new = result.x
                nrm = np.linalg.norm(c_new)
                if nrm > 1e-10:
                    c_new /= nrm
                R_new = _compute_ratio(c_new, basis_flat, W_flat)
                if R_new < best_R:
                    best_R = R_new
                    best_c = c_new.copy()
        except Exception:
            pass
    
    # Validate
    best_c = np.asarray(best_c, dtype=np.float64)
    if best_c.shape != (21,):
        return np.ones(21, dtype=np.float64)
    if np.any(~np.isfinite(best_c)):
        return np.ones(21, dtype=np.float64)
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3 or max_c > 1e5:
        return np.ones(21, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
