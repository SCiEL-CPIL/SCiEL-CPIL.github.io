# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def rudin_problem10() -> np.ndarray:
    """
    Finds coefficients for a degree-10 spherical harmonic that minimizes
    the L4/L2 norm ratio.
    """
    k = 10
    dim = 2 * k + 1  # 21
    
    # Quadrature setup
    # 70-point Gauss-Legendre in cos(theta)
    x, w = np.polynomial.legendre.leggauss(70)
    theta = np.arccos(x)
    
    # 130 equally spaced phi
    n_phi = 130
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    
    # Create grid
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    # Flatten
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    n_points = theta_flat.size
    
    # Weights for the sphere integral
    # Integral approx: sum_{i,j} (2*pi/n_phi) * w_i * f(theta_i, phi_j)
    # w_i are from leggauss (integrate over [-1,1] dx)
    # dOmega = dphi dx
    # So weight is (2*pi/n_phi) * w_i
    weights = np.kron(w, np.ones(n_phi)) * (2 * np.pi / n_phi)
    
    # Compute Spherical Harmonics Y_m at grid points
    # m ranges from -10 to 10
    # Y_0 = N_0 P_10^0
    # Y_m > 0: sqrt(2) N_m P_10^m cos(m phi)
    # Y_m < 0: sqrt(2) N_|m| P_10^|m| sin(|m| phi)
    
    # We need associated Legendre polynomials P_l^m(x) with Condon-Shortley phase.
    # SciPy has scipy.special.lpmv(m, l, x) which includes the Condon-Shortley phase (-1)^m.
    # The definition in the prompt: P^m = P_10^m(cos theta) with Condon-Shortley phase.
    # Standard scipy lpmv(m, l, x) returns P_l^m(x) with Condon-Shortley phase.
    
    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Note: The prompt defines N_m using |m| in the factorial for the denominator?
    # "N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)"
    # Usually N_m depends on |m|. Let's assume m in the formula refers to |m| for the factorial part,
    # or the formula is for m >= 0 and we use |m| for m < 0.
    # The prompt says "N_m = ... (10-m)!/(10+m)!". If m is negative, factorials of negative numbers are undefined.
    # So it must be |m|.
    
    def get_Y(m, theta, phi):
        abs_m = abs(m)
        # Factorial part
        # (10-abs_m)! / (10+abs_m)!
        # Using log-gamma to avoid overflow
        log_num = np.math.lgamma(10 - abs_m + 1)
        log_den = np.math.lgamma(10 + abs_m + 1)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(log_num - log_den))
        
        x = np.cos(theta)
        P_lm = scipy.special.lpmv(abs_m, 10, x)
        
        if m == 0:
            return N_m * P_lm
        elif m > 0:
            return np.sqrt(2.0) * N_m * P_lm * np.cos(m * phi)
        else:
            return np.sqrt(2.0) * N_m * P_lm * np.sin(abs_m * phi)

    Y_matrix = np.zeros((n_points, dim))
    for idx, m in enumerate(range(-10, 11)):
        Y_matrix[:, idx] = get_Y(m, theta_flat, phi_flat)
        
    # Precompute Y^T W for gradient calculations
    # grad I2 = 2 Y^T (W * u)
    # grad I4 = 4 Y^T (W * u^3)
    # We can precompute YT = Y.T
    YT = Y_matrix.T
    
    def compute_J_and_grad(c):
        u = Y_matrix @ c
        u2 = u * u
        u3 = u2 * u
        I2 = np.sum(weights * u2)
        I4 = np.sum(weights * u3 * u) # u^4
        
        if I2 == 0:
            return np.inf, np.zeros_like(c)
            
        J = I4 / (I2 * I2)
        
        # Gradient
        # dJ/dc = 4/I2^2 * Y^T (W * u^3 - (I4/I2) * W * u)
        term = weights * (u3 - (I4 / I2) * u)
        grad = (4.0 / (I2 * I2)) * (YT @ term)
        
        return J, grad

    best_c = np.ones(dim)
    best_J = np.inf
    
    # Optimization loop
    start_time = time.time()
    time_limit = 500.0 # Safety margin
    
    # We will run several restarts
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        if r == 0:
            c = np.ones(dim)
        else:
            c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        
        # Gradient Descent with line search
        c_curr = c.copy()
        J_curr, _ = compute_J_and_grad(c_curr)
        
        max_iters = 1000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            J_val, grad = compute_J_and_grad(c_curr)
            
            if not np.isfinite(J_val):
                break
                
            if J_val < best_J:
                best_J = J_val
                best_c = c_curr.copy()
                
            # Check convergence
            if np.linalg.norm(grad) < 1e-10:
                break
                
            # Backtracking line search
            alpha = 0.1
            for _ in range(20):
                c_new = c_curr - alpha * grad
                # Project to sphere
                norm_new = np.linalg.norm(c_new)
                if norm_new == 0:
                    break
                c_new /= norm_new
                
                J_new, _ = compute_J_and_grad(c_new)
                
                if J_new < J_curr:
                    c_curr = c_new
                    J_curr = J_new
                    break
                else:
                    alpha *= 0.5
            else:
                # If no step accepted, break
                break
                
    return best_c

# EVOLVE-BLOCK-END
