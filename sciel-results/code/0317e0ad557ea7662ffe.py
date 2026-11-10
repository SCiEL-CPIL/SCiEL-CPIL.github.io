# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
from scipy.optimize import differential_evolution
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, weights):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    # Normalize coeffs to unit norm to handle scale invariance and keep values stable
    norm_c = np.linalg.norm(coeffs)
    if norm_c < 1e-12:
        return 1e10
    c_norm = coeffs / norm_c
    
    f = Y @ c_norm
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 1e-15:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    # ||f||_4 = (sum w |f|^4)^(1/4)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_gl = len(x_gl)
    n_phi = len(phi)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    def objective(coeffs):
        # coeffs are in the bounded box [-1, 1]
        # We normalize them inside _compute_ratio
        return _compute_ratio(coeffs, Y, weights)

    # Define bounds for Differential Evolution
    # Since the ratio is scale invariant, we can search in a box
    # and normalize. The "shape" is what matters.
    bounds = [(-1.0, 1.0)] * 21
    
    # Differential Evolution parameters
    # popsize=20, mutation=(0.5, 1.0), recombination=0.9
    # maxiter=200, tol=1e-10
    # seed=42
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-10, 
            mutation=(0.5, 1.0), 
            recombination=0.9,
            polish=True, # Use L-BFGS-B to polish the result
            polish_kwargs={'maxiter': 100}
        )
        
        best_coeffs = res.x
        best_val = res.fun
        
    except Exception as e:
        print(f"Differential Evolution failed: {e}")
        # Fallback to a simple heuristic: m=0
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0
        best_val = _compute_ratio(best_coeffs, Y, weights)

    # If time allows, try a second DE run with different seed for robustness
    if time.time() < deadline - 60.0:
        try:
            res2 = differential_evolution(
                objective, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=20, 
                tol=1e-10, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                polish_kwargs={'maxiter': 50}
            )
            if res2.fun < best_val:
                best_coeffs = res2.x
                best_val = res2.fun
        except Exception as e:
            print(f"Second DE run failed: {e}")

    # Validate and normalize output
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        # Scale up if too small
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1.0
        else:
            return np.ones(21)
    elif max_abs > 1e5:
        # Scale down if too large
        best_coeffs = best_coeffs / max_abs * 1.0
        
    return best_coeffs

# EVOLVE-BLOCK-END
