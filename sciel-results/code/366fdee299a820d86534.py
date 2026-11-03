"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

# Precompute Hermite polynomials at quadrature nodes for speed
_N_QUAD = 100
_X_NODES, _X_WEIGHTS = np.polynomial.hermite.hermgauss(_N_QUAD)

def _precompute_basis():
    """
    Precompute the effective basis functions psi_0, psi_4, psi_8 
    that satisfy g(0)=0, and the corresponding Gram matrices A, B, C.
    """
    x = _X_NODES
    # Hermite polynomials
    H = {}
    H[0] = np.ones_like(x)
    H[1] = 2 * x
    for k in range(1, 12):
        H[k+1] = 2 * x * H[k] - 2 * k * H[k-1]
    
    # Derivatives H_n' = 2n H_{n-1}
    Hp = {}
    Hp[0] = np.zeros_like(x)
    for n in [4, 8, 12]:
        Hp[n] = 2 * n * H[n-1]
        
    # Values at 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 199584.0
    
    # Basis functions psi_i(x) = H_i(x) e^{-x^2/2} - (H_i(0)/H_12(0)) H_12(x) e^{-x^2/2}
    # Note: The Gaussian part e^{-x^2/2} is common to all, so we can include it in the basis.
    # Let phi_i(x) = H_i(x) * exp(-x^2/2)
    exp_term = np.exp(-x**2 / 2.0)
    
    phi_0 = H[0] * exp_term
    phi_4 = H[4] * exp_term
    phi_8 = H[8] * exp_term
    phi_12 = H[12] * exp_term
    
    # Derivatives of phi_i: (H_i' - x H_i) e^{-x^2/2}
    # We need psi_i' = (H_i' - x H_i) e^{-x^2/2} - (H_i(0)/H_12(0)) (H_12' - x H_12) e^{-x^2/2}
    
    def make_psi(phi_i, H_i_0, phi_12, Hp_12, H_12):
        # psi = phi_i - (H_i_0 / H_12_0) * phi_12
        coeff = H_i_0 / H12_0
        psi = phi_i - coeff * phi_12
        # psi' = phi_i' - coeff * phi_12'
        # phi_i' = (Hp[i] - x * H[i]) * exp_term
        # We don't have Hp for 0,4,8 stored in Hp dict, only 4,8,12. Let's compute on the fly or store all.
        # Actually, Hp[n] = 2n H[n-1].
        # Hp[0] = 0
        # Hp[4] = 8 H[3]
        # Hp[8] = 16 H[7]
        # Hp[12] = 24 H[11]
        
        # Let's just compute phi_i' directly
        # We need H[3] and H[7]
        # H is computed up to 12, so H[3], H[7], H[11] exist.
        return psi, (Hp[i] - x * H[i]) * exp_term if i in Hp else (2*i*H[i-1] - x * H[i]) * exp_term

    # Let's be more careful with derivatives
    # phi_i' = (H_i' - x H_i) e^{-x^2/2}
    
    psi_0 = phi_0 - (H0_0 / H12_0) * phi_12
    psi_4 = phi_4 - (H4_0 / H12_0) * phi_12
    psi_8 = phi_8 - (H8_0 / H12_0) * phi_12
    
    # Derivatives
    # H_0' = 0
    # H_4' = 8 H_3
    # H_8' = 16 H_7
    # H_12' = 24 H_11
    
    psi_0_prime = (0 - x * H[0]) * exp_term - (H0_0 / H12_0) * (24 * H[11] - x * H[12]) * exp_term
    psi_4_prime = (8 * H[3] - x * H[4]) * exp_term - (H4_0 / H12_0) * (24 * H[11] - x * H[12]) * exp_term
    psi_8_prime = (16 * H[7] - x * H[8]) * exp_term - (H8_0 / H12_0) * (24 * H[11] - x * H[12]) * exp_term
    
    # Basis arrays
    # Indices 0, 1, 2 correspond to c0, c4, c8
    basis = np.vstack([psi_0, psi_4, psi_8])
    basis_prime = np.vstack([psi_0_prime, psi_4_prime, psi_8_prime])
    
    # Compute Gram matrices
    # A_ij = <psi_i, psi_j> = sum w_k psi_i(k) psi_j(k)
    # B_ij = <x^2 psi_i, psi_j> = sum w_k x_k^2 psi_i(k) psi_j(k)
    # C_ij = <psi_i', psi_j'> = sum w_k psi_i'(k) psi_j'(k)
    
    # Using matrix multiplication for speed
    # A = basis @ diag(weights) @ basis.T
    # B = basis @ diag(weights * x^2) @ basis.T
    # C = basis_prime @ diag(weights) @ basis_prime.T
    
    W = _X_WEIGHTS
    X2 = _X_NODES**2
    
    A = (basis * W) @ basis.T
    B = (basis * W * X2) @ basis.T
    C = (basis_prime * W) @ basis_prime.T
    
    return A, B, C

_A, _B, _C = _precompute_basis()

def _objective(c):
    """
    Compute the uncertainty product bound using precomputed matrices.
    c = [c0, c4, c8]
    """
    # Handle potential numerical issues if c is zero
    norm_sq = c @ _A @ c
    if norm_sq < 1e-15:
        return 1e10
    
    x2_norm_sq = c @ _B @ c
    deriv_norm_sq = c @ _C @ c
    
    # Ensure non-negative (numerical noise can make these slightly negative)
    if x2_norm_sq < 0: x2_norm_sq = 0
    if deriv_norm_sq < 0: deriv_norm_sq = 0
    
    # Objective: (dx^2) * (dxi^2)
    # dx^2 = x2_norm_sq / norm_sq
    # dxi^2 = deriv_norm_sq / (4 * pi^2 * norm_sq)
    val = (x2_norm_sq * deriv_norm_sq) / (4.0 * np.pi**2 * norm_sq**2)
    
    return val

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product bound.
    Uses Differential Evolution for global search, then local polish.
    """
    start_time = time.time()
    deadline = start_time + 190  # Safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Check seed value to ensure consistency
    seed_val = _objective(seed)
    
    # Global Optimization
    # Bounds: The coefficients can be scaled arbitrarily, so we need to constrain the search space.
    # However, DE works best with bounded variables.
    # We can normalize the vector c to have norm 1 in some metric, or just bound it.
    # Since the function is homogeneous, the magnitude doesn't matter, only the direction.
    # We can fix c8 = 1 and search c0, c4? No, c8 could be 0.
    # Let's bound the coefficients to a reasonable range, e.g., [-10, 10].
    # If the optimum is outside, we might miss it, but given the seed is ~[-1.13, 0.04, 0.0003],
    # the scale is roughly 1.
    
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Run Differential Evolution
    if time.time() - start_time < deadline - 30:
        try:
            res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception as e:
            pass
            
    # Local Polish with L-BFGS-B
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
    # Additional local restarts if time permits
    if time.time() - start_time < deadline - 5:
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() - start_time > deadline - 2:
                break
            # Perturb best solution
            x0 = best_coeffs + rng.normal(0, 0.1, size=3)
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass

    if time.time() - start_time > deadline:
        return seed

    return best_coeffs
# EVOLVE-BLOCK-END
