# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using Simulated Annealing.
    Returns the optimal coefficients c_m for m = -10..10.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # Map from [-1, 1] to theta in [0, pi]
    # x = cos(theta), so theta = arccos(x)
    # The weight for surface measure dOmega = sin(theta) dtheta dphi
    # dtheta = -dx / sin(theta)
    # So sin(theta) dtheta = -dx
    # The integral over theta is from 0 to pi, so x goes from 1 to -1.
    # Standard Gauss-Legendre integrates from -1 to 1.
    # So we just need the weights w_gl directly for the x-integral.
    # The total weight sum should be 2.
    
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Grid points
    # We need to evaluate Y_m(theta, phi)
    # theta is derived from x_gl. 
    # Note: x_gl are in [-1, 1]. Let's call them cos_theta_vals.
    cos_theta_vals = x_gl
    sin_theta_vals = np.sqrt(np.maximum(0, 1 - cos_theta_vals**2))
    
    # 2. Pre-compute Basis Functions Y_m
    # Y_0 = N_0 P_10^0
    # Y_m = sqrt(2) N_m P_10^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_m P_10^|m| sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    
    # We store Y_m values at all grid points in a matrix of shape (n_coeffs, n_theta, n_phi)
    # Or more efficiently, flatten the grid to (n_theta * n_phi,)
    n_grid = 70 * 130
    
    # Initialize Y matrix
    Y = np.zeros((n_coeffs, n_grid), dtype=np.float64)
    
    # Precompute factorial ratios for normalization
    # N_m = sqrt( (21)/(4*pi) * (10-m)! / (10+m)! )
    # Using log-factorials to avoid overflow/underflow issues, though 10! is small.
    # 10! = 3,628,800. No overflow.
    
    for m_idx in range(n_coeffs):
        m = m_idx - k
        
        # Calculate N_m
        # (10-m)! / (10+m)!
        num = factorial(10 - abs(m))
        den = factorial(10 + abs(m))
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        # Calculate Associated Legendre Polynomial P_10^|m|(cos_theta)
        # scipy.special.lpmv(m, l, x) includes Condon-Shortley phase (-1)^m
        # The problem statement specifies "with the Condon-Shortley phase", 
        # which lpmv provides.
        P_vals = lpmv(abs(m), 10, cos_theta_vals)
        
        # P_vals shape: (70,)
        # We need to expand this to the full grid (70, 130) and apply phi dependence.
        
        if m == 0:
            # Y_0 = N_0 P_10^0
            # Broadcast P_vals to (70, 130)
            Y_vals = np.tile(P_vals[:, np.newaxis], (1, n_phi)) * N_m
        elif m > 0:
            # Y_m = sqrt(2) N_m P_10^m cos(m phi)
            cos_term = np.cos(m * phi) # shape (130,)
            # P_vals[:, np.newaxis] * cos_term[np.newaxis, :]
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * cos_term[np.newaxis, :])
        else:
            # m < 0
            abs_m = -m
            # Y_m = sqrt(2) N_m P_10^|m| sin(|m| phi)
            sin_term = np.sin(abs_m * phi) # shape (130,)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * sin_term[np.newaxis, :])
        
        # Flatten and store
        Y[m_idx, :] = Y_vals.flatten()
        
    # 3. Define Objective Function
    # R = ||f||_4 / ||f||_2
    # We want to MINIMIZE R.
    # The evaluator measures 1/R (higher is better).
    # So we minimize R, which is equivalent to maximizing 1/R.
    
    def compute_R(coeffs):
        # coeffs shape (21,)
        # f = sum c_m Y_m
        # Y shape (21, n_grid)
        f_vals = Y.T @ coeffs # shape (n_grid,)
        
        # Quadrature weights
        # The surface integral is sum_i w_i f(x_i)
        # The weight for each grid point (theta_i, phi_j) is w_gl_i * dphi
        # Note: The Gauss-Legendre weights w_gl integrate dx from -1 to 1.
        # The surface element is sin(theta) dtheta dphi = -dx dphi.
        # So the integral is \int_{-1}^1 \int_0^{2pi} F(x, phi) dphi dx.
        # Discretized: sum_i sum_j F(x_i, phi_j) w_gl_i dphi
        
        weights = np.tile(w_gl, (n_phi,)) * dphi
        weights = weights.flatten()
        
        # ||f||_2^2 = \int |f|^2 dOmega
        # ||f||_4^4 = \int |f|^4 dOmega
        
        f_sq = f_vals**2
        f_four = f_sq**2
        
        norm2_sq = np.sum(weights * f_sq)
        norm4_four = np.sum(weights * f_four)
        
        # Handle zero norm case (should not happen if coeffs are non-zero)
        if norm2_sq < 1e-15:
            return np.inf
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_four)) # ||f||_4 = (||f||_4^4)^(1/4)
        
        return norm4 / norm2

    # 4. Simulated Annealing Optimization
    start_time = time.time()
    time_limit = 600.0 # seconds
    
    # Initial guess: All ones (same as seed) or random
    # Let's start with ones, normalized to unit L2 norm to stabilize scaling?
    # The ratio is scale-invariant. 
    # c -> alpha c => R(alpha c) = R(c).
    # So we can fix the scale, e.g., ||c||_2 = 1 or sum c^2 = 1.
    # But the problem asks for coefficients c. Any scale is valid.
    # Let's just work with the vector directly.
    
    c_current = np.ones(n_coeffs, dtype=np.float64)
    # Normalize to prevent numerical issues if values grow/shrink
    c_current /= np.linalg.norm(c_current)
    
    R_current = compute_R(c_current)
    c_best = c_current.copy()
    R_best = R_current
    
    # Annealing parameters
    T = 0.1 # Initial temperature
    cooling_rate = 0.995
    T_min = 1e-6
    
    # Step size
    step_scale = 0.1
    
    while time.time() - start_time < time_limit - 10.0: # 10s safety margin
        # Generate perturbation
        # Gaussian noise
        noise = np.random.normal(0, step_scale, n_coeffs)
        c_new = c_current + noise
        
        # Optional: Re-normalize to keep vector magnitude sane?
        # Since R is scale invariant, we can project back to unit sphere in coefficient space
        # to avoid drifting to huge/small values which might cause float issues.
        norm_c_new = np.linalg.norm(c_new)
        if norm_c_new < 1e-10:
            c_new = c_current
        else:
            c_new = c_new / norm_c_new
            
        R_new = compute_R(c_new)
        
        # Acceptance criterion
        if R_new < R_current:
            # Improvement
            c_current = c_new
            R_current = R_new
            
            if R_new < R_best:
                R_best = R_new
                c_best = c_new.copy()
        else:
            # Worse, accept with probability exp(-(R_new - R_current)/T)
            delta_R = R_new - R_current
            if delta_R < 0: # Should not happen in else branch
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_R / T)
                
            if np.random.random() < accept_prob:
                c_current = c_new
                R_current = R_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            # Restart with higher temperature? Or just stop?
            # With 600s, we can run many iterations.
            # Let's restart T to explore more.
            T = 0.05
            # Maybe reduce step size slightly?
            step_scale *= 0.95
            if step_scale < 1e-4:
                step_scale = 1e-4

    # Final check for validity
    # Invalid: size not 21, NaN or inf, max|c| < 1e-3 or max|c| > 1e5.
    # Our c_best is normalized to unit norm, so max|c| <= 1.
    # min|c| could be 0, but max|c| will be > 0.
    # The constraint is max|c| < 1e-3 (too small) or > 1e5 (too big).
    # Since norm is 1, max|c| is between 1/sqrt(21) ~ 0.2 and 1.
    # So it is always valid regarding magnitude bounds.
    
    # Ensure no NaNs
    if not np.all(np.isfinite(c_best)):
        return np.ones(n_coeffs, dtype=np.float64)
        
    return c_best

# EVOLVE-BLOCK-END
