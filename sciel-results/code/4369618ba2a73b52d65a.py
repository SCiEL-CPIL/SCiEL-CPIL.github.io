# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from math import lgamma, log, pi

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi)
Y_basis = np.zeros((21, n_theta, n_phi))
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl)
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Weights for integration
# Integral approx = sum_i w_gl[i] * (2pi/n_phi) * sum_j f[i,j]
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()

# Reshape Y_basis to (21, 70*130)
Y_flat = Y_basis.reshape(21, -1)
N_POINTS = 70 * 130

def evaluate_ratio(coeffs):
    """
    Computes R = ||f||_4 / ||f||_2 for given coefficients.
    """
    f_flat = coeffs @ Y_flat
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def gradient_descent_on_sphere(c0, max_iter=200, tol=1e-8, time_limit=None):
    """
    Minimize J(c) = ||f(c)||_4^4 subject to ||c||_2 = 1.
    Since R = ||f||_4 / ||f||_2, and ||f||_2 is constant if ||c||_2=1 (due to orthonormality),
    minimizing R is equivalent to minimizing ||f||_4.
    """
    c = c0.copy()
    c_norm = np.linalg.norm(c)
    if c_norm == 0:
        c = np.random.randn(21)
        c_norm = np.linalg.norm(c)
    c = c / c_norm
    
    best_c = c.copy()
    best_val = np.sum(W_flat * (c @ Y_flat)**4)
    
    # Adaptive step size
    alpha = 1.0
    last_val = best_val
    
    for i in range(max_iter):
        if time_limit is not None:
            if time.time() > time_limit:
                break
        
        f_flat = c @ Y_flat
        # Gradient of J(c) = sum w_i f_i^4 is 4 * Y^T (w * f^3)
        grad = 4.0 * (Y_flat.T @ (W_flat * f_flat**3))
        
        # Project gradient onto tangent space of sphere at c
        # grad_tangent = grad - (grad . c) * c
        proj = np.dot(grad, c)
        grad_tangent = grad - proj * c
        
        # Line search / Step size adjustment
        # Try a step
        c_new = c - alpha * grad_tangent
        c_new_norm = np.linalg.norm(c_new)
        if c_new_norm < 1e-10:
            alpha *= 0.5
            continue
        c_new = c_new / c_new_norm
        
        val_new = np.sum(W_flat * (c_new @ Y_flat)**4)
        
        if val_new < best_val:
            best_val = val_new
            best_c = c_new.copy()
            # If improvement, maybe increase alpha slightly? Or keep.
            # Standard: if step was good, keep alpha. If bad, reduce.
            c = c_new
            last_val = val_new
        else:
            # Reduce step size
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        # Convergence check
        if abs(last_val - val_new) < tol * max(1.0, abs(val_new)):
             if val_new < best_val:
                 best_val = val_new
                 best_c = c_new.copy()
             break
             
    return best_c, best_val

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
    start_time = time.time()
    max_time = 600.0
    
    # Strategy: Multi-start Gradient Descent on the Unit Sphere
    # We minimize ||f||_4^4 which is proportional to R^4 * ||f||_2^4.
    # Since we fix ||c||_2=1, ||f||_2 is constant (sqrt(sum c^2) * ||Y||_2... wait).
    # The basis Y_m is orthonormal. So ||f||_2^2 = sum c_m^2 * ||Y_m||_2^2 = sum c_m^2 * 1 = 1.
    # So minimizing R is exactly minimizing ||f||_4.
    
    best_coeffs = None
    best_ratio = np.inf
    
    # Generate diverse starting points
    # 1. Random uniform on sphere
    # 2. Sparse starts (few non-zero coeffs)
    # 3. Structured starts (e.g., only m=0, or m=10, etc.)
    
    starts = []
    
    # Random starts
    for _ in range(20):
        c = np.random.randn(21)
        c /= np.linalg.norm(c)
        starts.append(c)
        
    # Sparse starts
    for _ in range(10):
        c = np.zeros(21)
        n_nonzero = np.random.randint(2, 6)
        indices = np.random.choice(21, n_nonzero, replace=False)
        c[indices] = np.random.randn(n_nonzero)
        c /= np.linalg.norm(c)
        starts.append(c)
        
    # Structured: Single mode
    for m_idx in range(21):
        c = np.zeros(21)
        c[m_idx] = 1.0
        starts.append(c)
        
    # Structured: Pairs of modes (interference)
    for _ in range(10):
        c = np.zeros(21)
        i, j = np.random.choice(21, 2, replace=False)
        c[i] = 1.0
        c[j] = np.random.choice([-1, 1])
        c /= np.linalg.norm(c)
        starts.append(c)

    # Evaluate all starts with Gradient Descent
    # We have ~40 starts. Each GD run should be fast.
    # Time budget per start: (max_time - 10) / len(starts)
    time_per_start = (max_time - 10.0) / len(starts)
    
    for i, c0 in enumerate(starts):
        if time.time() - start_time > max_time - 5.0:
            break
            
        deadline = time.time() + time_per_start
        c_opt, val_opt = gradient_descent_on_sphere(c0, max_iter=100, time_limit=deadline)
        
        # Calculate actual ratio for the optimized coefficients
        ratio = evaluate_ratio(c_opt)
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_coeffs = c_opt.copy()
            
    # If we still have time, do a few more random restarts from the best found
    if best_coeffs is not None and time.time() - start_time < max_time - 10.0:
        for _ in range(5):
            if time.time() - start_time > max_time - 5.0:
                break
            noise = np.random.randn(21) * 0.1
            c0 = best_coeffs + noise
            c0 /= np.linalg.norm(c0)
            c_opt, _ = gradient_descent_on_sphere(c0, max_iter=50, time_limit=time.time() + 10.0)
            ratio = evaluate_ratio(c_opt)
            if ratio < best_ratio:
                best_ratio = ratio
                best_coeffs = c_opt.copy()

    # Final Validation
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Scale to ensure max|c| is within [1e-3, 1e5]
    # The ratio is scale invariant, so we can scale freely.
    # Let's scale to max|c| = 1.0 for cleanliness, unless it's 0.
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs = best_coeffs * (1.0 / max_c) if max_c > 0 else best_coeffs
    elif max_c > 1e5:
        best_coeffs = best_coeffs / max_c * 1.0
    else:
        # Keep as is, or normalize to 1 for consistency? 
        # The problem doesn't require a specific scale, just valid range.
        # Normalizing to 1.0 is safe.
        if max_c > 0:
             best_coeffs = best_coeffs / max_c * 1.0

    return best_coeffs

# EVOLVE-BLOCK-END
