import numpy as np
import time
from scipy.special import lpmv, factorial
import warnings
warnings.filterwarnings('ignore')

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
    that MINIMIZE the norm ratio ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = 630  # seconds
    
    # Parameters
    l = 10
    n_coeffs = 2 * l + 1
    
    # Quadrature settings
    # Prompt suggests 70 GL points and 130 phi points. 
    # Let's use slightly more for safety against any floating point accumulation, 
    # though 70x130 is theoretically exact.
    N_gl = 70
    N_phi = 130
    
    x_gl, w_gl = np.polynomial.legendre.leggauss(N_gl)
    phi_grid = 2 * np.pi * np.arange(N_phi) / N_phi
    dphi = 2 * np.pi / N_phi
    
    # Flatten grids
    # We want to construct M such that ||M c||_2^2 = ||f||_2^2 and ||M c||_4^4 = ||f||_4^4
    # M has shape (N_points, n_coeffs)
    # M[i, j] = sqrt(w_i * dphi) * Y_j(theta_i, phi_i)
    
    # Precompute Y values
    # theta = arccos(x_gl)
    # We can compute P_l^m(x_gl) for all m
    # P_l^m(x) with Condon-Shortley phase is given by lpmv(m, l, x)
    
    Y_matrix = np.zeros((N_gl * N_phi, n_coeffs))
    
    for idx, m in enumerate(range(-l, l + 1)):
        abs_m = abs(m)
        # Compute P_l^m(x_gl)
        P = lpmv(abs_m, l, x_gl)
        
        # Normalization constant N_m
        # N_m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
        n_fact = factorial(l - abs_m, exact=True)
        d_fact = factorial(l + abs_m, exact=True)
        N_m = np.sqrt((2 * l + 1) / (4 * np.pi) * (n_fact / d_fact))
        
        # Tile P for all phi points
        P_flat = np.tile(P, N_phi)
        
        if m == 0:
            Y_val = N_m * P_flat
        elif m > 0:
            # Y_m = sqrt(2) N_m P^m cos(m phi)
            # phi_flat = tile(phi_grid, N_gl)
            phi_flat = np.tile(phi_grid, N_gl)
            Y_val = np.sqrt(2) * N_m * P_flat * np.cos(m * phi_flat)
        else:
            # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi)
            phi_flat = np.tile(phi_grid, N_gl)
            Y_val = np.sqrt(2) * N_m * P_flat * np.sin(abs_m * phi_flat)
            
        Y_matrix[:, idx] = Y_val
        
    # Apply quadrature weights to M
    # Weight for each point is w_gl[i] * dphi
    # We need M such that sum (M c)^2 = sum w_i dphi Y_i^2
    # So M should be sqrt(w_i dphi) * Y_i
    weights_flat = np.repeat(w_gl, N_phi) * dphi
    M = Y_matrix * np.sqrt(weights_flat)[:, None]
    
    # Now ||M c||_2^2 = ||f||_2^2
    # And ||M c||_4^4 = sum (M c)^4 = ||f||_4^4
    
    # Power method for nonlinear eigenvalue problem:
    # Minimize R = ||f||_4 / ||f||_2
    # Equivalent to minimizing R^2 = ||M c||_4^4 / ||M c||_2^2
    # Stationarity: M^T (M c)^3 = lambda c
    
    def power_method(c_init, max_iter=1000, tol=1e-12):
        c = c_init.copy()
        c = c / np.linalg.norm(c)
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Compute f = M c
            f = M @ c
            # Compute J(c) = M^T f^3
            Jc = M.T @ (f ** 3)
            
            # Normalize
            norm_Jc = np.linalg.norm(Jc)
            if norm_Jc < 1e-15:
                break
                
            c_new = Jc / norm_Jc
            
            # Check convergence
            # Use angle or difference
            diff = np.linalg.norm(c_new - c)
            c = c_new
            
            if diff < tol:
                break
                
        return c
    
    # Multi-start to ensure global minimum
    # The problem is symmetric, so the global min is unique up to sign.
    # Random starts should converge to it.
    
    rng = np.random.default_rng(42)
    
    best_c = None
    best_ratio = np.inf
    
    # Initial guesses
    # 1. Random
    for _ in range(20):
        c_init = rng.normal(size=n_coeffs)
        c_opt = power_method(c_init)
        
        # Compute ratio
        f = M @ c_opt
        norm2_sq = np.dot(f, f)
        norm4_4 = np.dot(f, f**3)
        
        if norm2_sq > 0:
            ratio = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
            if ratio < best_ratio:
                best_ratio = ratio
                best_c = c_opt.copy()
                
    # 2. Structured guesses (often help with convergence speed)
    # Single modes
    for i in range(n_coeffs):
        c_init = np.zeros(n_coeffs)
        c_init[i] = 1.0
        c_opt = power_method(c_init)
        f = M @ c_opt
        norm2_sq = np.dot(f, f)
        norm4_4 = np.dot(f, f**3)
        if norm2_sq > 0:
            ratio = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
            if ratio < best_ratio:
                best_ratio = ratio
                best_c = c_opt.copy()

    # 3. Pairs
    for i in range(n_coeffs):
        for j in range(i+1, n_coeffs):
            c_init = np.zeros(n_coeffs)
            c_init[i] = 1.0
            c_init[j] = 1.0
            c_opt = power_method(c_init)
            f = M @ c_opt
            norm2_sq = np.dot(f, f)
            norm4_4 = np.dot(f, f**3)
            if norm2_sq > 0:
                ratio = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_c = c_opt.copy()

    if best_c is None:
        best_c = np.ones(n_coeffs)
        
    # Final check and scaling
    # The problem does not require a specific scale, but we should ensure it's valid.
    # "Invalid: ... max|c| < 1e-3 or max|c| > 1e5"
    # We can normalize to max abs value = 1.0
    
    max_abs = np.max(np.abs(best_c))
    if max_abs > 0:
        best_c = best_c / max_abs
        
    return best_c.astype(np.float64)
