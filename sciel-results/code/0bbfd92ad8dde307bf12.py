import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = 600  # Safety margin
    
    k = 10
    num_coeffs = 2 * k + 1
    
    # 1. Setup Quadrature
    # 70-point Gauss-Legendre in cos(theta)
    x, w = np.polynomial.legendre.leggauss(70)
    # x is cos(theta), w are weights.
    # We need to map to theta and phi.
    # Surface element dOmega = sin(theta) dtheta dphi.
    # Integral = sum w_i * sum_j f(theta_i, phi_j) * dphi.
    # Since x = cos(theta), dx = -sin(theta) dtheta.
    # So sum w_i is the integral over cos(theta) from -1 to 1.
    # The weights w_i already account for the Jacobian of the variable change x=cos(theta)?
    # leggauss returns nodes and weights for integral_{-1}^1 g(x) dx.
    # Here g(x) = \int f(arccos(x), phi) dphi.
    # So yes, the weights w_i are correct for the \cos \theta integration.
    
    # 130 equally spaced phi
    num_phi = 130
    phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    dphi = 2 * np.pi / num_phi
    
    # Total weights for each (i, j) point
    # W_{i,j} = w_i * dphi
    # We can flatten this.
    # Total points = 70 * 130 = 9100
    
    # Precompute Basis Functions
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^{|m|}(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    
    # Factorials for N_m
    # (10-m)! / (10+m)!
    # We can compute log factorials or just use math.factorial for small numbers.
    fact = np.array([1.0] * 21)
    for i in range(1, 21):
        fact[i] = fact[i-1] * i
        
    # Precompute N_m for m = 0..10
    N = np.zeros(11)
    for m in range(11):
        # (10-m)! / (10+m)!
        # Note: fact[10-m] / fact[10+m]
        ratio = fact[10-m] / fact[10+m]
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
        
    # Create arrays for grid
    # theta_i from x_i = cos(theta_i)
    # We need sin(m phi) and cos(m phi)
    
    # Flatten grid indices
    # i in 0..69, j in 0..129
    # Total N_pts = 70 * 130
    N_pts = 70 * 130
    
    # We will store Y_m for each m in a matrix of shape (N_pts, 21)
    # Or compute on the fly? Precomputing is faster.
    # Let's precompute the "shape" part P_10^m(x_i) and the "phase" part.
    
    # P_10^m(x) for x in x (70 points)
    # lpmv(m, 10, x) computes P_10^m(x) with Condon-Shortley phase.
    
    P_vals = np.zeros((11, 70))
    for m in range(11):
        P_vals[m] = lpmv(m, 10, x)
        
    # Now construct Y_m at all 9100 points
    # Y_m = coeff_m * P_m(x_i) * phase(m, phi_j)
    
    # We can vectorize this.
    # x is shape (70,)
    # phi is shape (130,)
    # P_vals[m] is shape (70,)
    
    # Let's create the full basis matrix B of shape (N_pts, 21)
    # B[:, idx] = Y_{m_idx}
    
    # We can expand P_vals to (70, 130) by repeating columns?
    # P_expanded[m, i, j] = P_vals[m, i]
    
    # Actually, let's just compute the objective function efficiently.
    # f_i = sum_m c_m Y_m(x_i, phi_j)
    
    # Let's precompute the "angular" parts for each m.
    # For m > 0: cos(m phi_j)
    # For m < 0: sin(|m| phi_j)
    # For m = 0: 1
    
    # We will store the basis functions in a list or array.
    # Basis[m_offset] has shape (N_pts,)
    
    Basis = np.zeros((N_pts, 21))
    
    # Expand x to (70, 130)
    # x_grid = np.repeat(x[:, None], 130, axis=1) -> shape (70, 130)
    # phi_grid = np.tile(phi[None, :], (70, 1)) -> shape (70, 130)
    
    # This might use too much memory? 70*130 is small.
    # Let's do it.
    
    x_grid = np.repeat(x[:, None], 130, axis=1) # (70, 130)
    phi_grid = np.tile(phi[None, :], (70, 1))   # (70, 130)
    
    # Flatten
    x_flat = x_grid.flatten() # (9100,)
    phi_flat = phi_grid.flatten() # (9100,)
    
    # Weights
    # w_i is for each row i.
    # w_grid = np.repeat(w[:, None], 130, axis=1)
    # w_flat = w_grid.flatten() * dphi
    w_flat = np.repeat(w * dphi, 130) # (9100,)
    
    # Compute Basis
    for m_idx in range(21):
        m = m_idx - 10
        abs_m = abs(m)
        
        # P_10^|m|(x_flat)
        P_part = P_vals[abs_m][np.repeat(np.arange(70), 130)] # (9100,)
        
        if m == 0:
            phase_part = 1.0
            coeff = N[0]
        elif m > 0:
            phase_part = np.cos(m * phi_flat)
            coeff = np.sqrt(2.0) * N[m]
        else:
            phase_part = np.sin(abs_m * phi_flat)
            coeff = np.sqrt(2.0) * N[abs_m]
            
        Basis[:, m_idx] = coeff * P_part * phase_part
        
    # Objective Function
    def objective(c):
        # c is shape (21,)
        # f = Basis @ c
        f = Basis @ c
        
        # ||f||_2^2 = sum w_i f_i^2
        norm2_sq = np.sum(w_flat * f**2)
        if norm2_sq < 1e-30:
            return 1e100
            
        # ||f||_4^4 = sum w_i |f_i|^4
        norm4_4 = np.sum(w_flat * np.abs(f)**4)
        
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sqrt(np.sqrt(norm4_4))
        
        return norm4 / norm2
        
    # Multi-start optimization
    best_c = np.ones(21)
    best_val = objective(best_c)
    
    # Random restarts
    # Use fixed seed for reproducibility in evaluation? No, random is fine.
    # But we want to be fast.
    
    num_restarts = 100
    # Estimate time per restart.
    # L-BFGS-B with 21 vars, ~9100 points.
    # Should be very fast.
    
    # Let's use Nelder-Mead or L-BFGS-B.
    # L-BFGS-B is better for smooth functions.
    
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Random initialization
        # Normal distribution
        c_init = np.random.randn(21)
        
        # Try to optimize
        try:
            res = minimize(objective, c_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
        except Exception:
            pass
            
    # Final check
    if time.time() > deadline - 10:
        return best_c
        
    # Maybe one more pass with the best result to polish?
    # Or just return.
    
    # Ensure valid range
    # max|c| < 1e-3 or > 1e5 is invalid.
    # The optimizer might produce very small or large values?
    # The function is homogeneous, so scale doesn't matter for R.
    # But the validator checks the magnitude of c.
    # "Invalid: ... max|c| < 1e-3 or max|c| > 1e5"
    # We should normalize best_c to have a reasonable magnitude, e.g., max|c| = 1.
    
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
    else:
        # Normalize to 1 for consistency, doesn't change R
        best_c = best_c / max_c
        
    return best_c
