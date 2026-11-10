# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
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

def _compute_ratio(coeffs, Y, w_gl, dphi):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs  # shape (n_points,)
    n_gl = len(w_gl)
    n_phi = Y.shape[0] // n_gl
    weights = np.repeat(w_gl, n_phi) * dphi
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(norm4_sq)
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    n_points = Y.shape[0]
    
    # Precompute weights
    n_gl = len(w_gl)
    n_phi = Y.shape[0] // n_gl
    weights = np.repeat(w_gl, n_phi) * dphi
    
    # Objective: minimize ||f||_4 / ||f||_2
    # This is equivalent to minimizing (||f||_4 / ||f||_2)^2 = ||f||_4^2 / ||f||_2^2
    # For a fixed direction, the ratio is scale-invariant, so we can fix ||f||_2 = 1
    # and minimize ||f||_4^2.
    # 
    # Key insight: The ratio ||f||_4/||f||_2 is minimized when f is as "flat" as possible.
    # For spherical harmonics of degree 10, the flattest harmonic should have significant
    # contribution from multiple m values to spread the energy.
    #
    # Strategy: Use a power-iteration-like approach in the space of normalized functions.
    # Since the ratio is scale-invariant, we can think of it as:
    #   R = ||f||_4 / ||f||_2
    # Minimizing R is equivalent to finding the direction that minimizes R.
    #
    # Alternative approach: Since R^2 = ||f||_4^2 / ||f||_2^2, and both are homogeneous,
    # we can fix ||f||_2^2 = 1 (i.e., sum w*f^2 = 1) and minimize ||f||_4^2 = sum w*f^4.
    #
    # Let's use a different strategy: exploit the fact that for degree-10 harmonics,
    # the optimal "flattest" harmonic likely has symmetry. We'll search over a reduced
    # parameter space using the structure of the harmonics.
    #
    # Actually, let's use a more direct approach: the ratio R = ||f||_4/||f||_2 depends
    # only on the direction of f in the 21D coefficient space. So we can parameterize
    # by angles on the 20-sphere. But that's still 20D.
    #
    # Better: Use the fact that we can normalize to ||f||_2 = 1 and then minimize ||f||_4^2.
    # This is a constrained optimization: minimize sum(w_i * f_i^4) subject to sum(w_i * f_i^2) = 1.
    # By Lagrange multipliers, at optimum: 4 w_i f_i^3 = lambda * 2 w_i f_i, so f_i^2 = lambda/2
    # for all i where f_i != 0. This means at the optimum, |f_i| is constant wherever f_i != 0.
    # This is the "eigenfunction" insight!
    #
    # So the optimal f should have |f(x)| = constant on the set where f(x) != 0.
    # For a spherical harmonic, this means it should be as "flat" as possible.
    #
    # Practical approach: Use iterative normalization. Start with a candidate,
    # then project it toward the "flat" direction.
    #
    # Let's try: minimize R by gradient descent on the normalized coefficients.
    # The gradient of R = ||f||_4/||f||_2 with respect to coeffs is:
    # dR/dc = (1/||f||_2) * d||f||_4/dc - (||f||_4/||f||_2^2) * d||f||_2/dc
    # d||f||_4/dc = (1/(2||f||_4)) * 4 * sum(w_i * f_i^3 * Y_i) = (2/||f||_4) * (Y^T * (w * f^3))
    # d||f||_2/dc = (1/||f||_2) * 2 * sum(w_i * f_i * Y_i) = (2/||f||_2) * (Y^T * (w * f))
    #
    # So dR/dc = (2/||f||_4) * (Y^T (w * f^3)) / ||f||_2 - (||f||_4/||f||_2^2) * (2/||f||_2) * (Y^T (w * f))
    #          = (2/(||f||_4 * ||f||_2)) * (Y^T (w * f^3)) - (2 * ||f||_4 / ||f||_2^3) * (Y^T (w * f))
    
    def objective_and_grad(coeffs):
        f = Y @ coeffs
        f2 = f * f
        f4 = f2 * f2
        norm2_sq = np.dot(weights, f2)
        norm4_sq = np.dot(weights, f4)
        if norm2_sq <= 0:
            return 1e10, np.zeros_like(coeffs)
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(norm4_sq)
        R = norm4 / norm2
        
        # Gradient
        grad_term1 = Y.T @ (weights * f**3)
        grad_term2 = Y.T @ (weights * f)
        grad = (2.0 / (norm4 * norm2)) * grad_term1 - (2.0 * norm4 / norm2**3) * grad_term2
        
        return R, grad
    
    # Start with a good initial guess: mix of several m values
    # The flattest harmonic likely has contributions from m = 0, ±1, ±2, ...
    # Let's start with equal weights on all m values
    x = np.ones(2 * k + 1, dtype=np.float64)
    
    best_coeffs = x.copy()
    best_val = objective_and_grad(x)[0]
    
    # Gradient descent with adaptive step size
    lr = 0.01
    n_steps = 50000
    patience = 5000
    no_improve = 0
    
    for step in range(n_steps):
        if time.time() > deadline - 30.0:
            break
        
        val, grad = objective_and_grad(x)
        
        if val < best_val:
            best_val = val
            best_coeffs = x.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            # Reduce learning rate and reset patience
            lr *= 0.5
            no_improve = 0
            if lr < 1e-10:
                break
        
        # Gradient step
        x_new = x - lr * grad
        
        # Check validity
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            lr *= 0.1
            x_new = x - lr * grad
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                break
        
        x = x_new
    
    # Also try a few random restarts with different initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 60.0:
            break
        
        x = np.random.randn(2 * k + 1)
        # Normalize
        f = Y @ x
        norm2 = np.sqrt(np.dot(weights, f * f))
        if norm2 > 0:
            x = x / norm2
        
        lr = 0.01
        no_improve = 0
        
        for step in range(30000):
            if time.time() > deadline - 30.0:
                break
            
            val, grad = objective_and_grad(x)
            
            if val < best_val:
                best_val = val
                best_coeffs = x.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-10:
                    break
            
            x_new = x - lr * grad
            
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                lr *= 0.1
                x_new = x - lr * grad
                if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
                    break
            
            x = x_new
    
    # Validate
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
