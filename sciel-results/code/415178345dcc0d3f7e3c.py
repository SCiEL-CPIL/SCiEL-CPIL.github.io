import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
from scipy.integrate import gauss_legendre

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    L = 10
    n_m = 2 * L + 1  # 21 coefficients
    
    # 1. Setup Quadrature Grid
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    x, w = gauss_legendre(n_theta)
    # x are cos(theta) values in [-1, 1]
    # w are weights for integration over [-1, 1]
    
    # Phi values
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Create meshgrid
    # We want to evaluate Y_m at all (theta, phi)
    # cos_theta = x[:, None] * np.ones((1, n_phi))
    # sin_theta = sqrt(1 - x^2)
    # phi = np.ones((n_theta, 1)) * phi[None, :]
    
    cos_theta = np.repeat(x[:, None], n_phi, axis=1)
    sin_theta = np.sqrt(np.clip(1 - cos_theta**2, 0, 1))
    phi_grid = np.tile(phi[None, :], (n_theta, 1))
    
    # Weights for the 2D grid
    # dOmega = sin(theta) dtheta dphi? No, standard spherical coords:
    # dOmega = sin(theta) dtheta dphi.
    # Change of variable u = cos(theta) -> du = -sin(theta) dtheta.
    # So dOmega = dphi du.
    # The integral is sum over j,k w_j * d_phi * f(u_j, phi_k)
    # Total weight W_{jk} = w_j * d_phi
    W = np.tile(w[:, None], (1, n_phi)) * d_phi
    
    # Flatten weights for vectorized operations
    W_flat = W.flatten()
    N_grid = W_flat.size
    
    # 2. Precompute Spherical Harmonics Y_m on the grid
    # Y_m values for m = -10 .. 10
    # Y is shape (N_grid, 21)
    
    Y = np.zeros((N_grid, n_m), dtype=np.float64)
    
    # Normalization constants N_m
    # N_m = sqrt( (2L+1)/(4*pi) * (L-m)! / (L+m)! )
    # We need factorials. Since L=10, we can compute directly.
    def norm_const(m):
        m_abs = abs(m)
        # (10 - m_abs)! / (10 + m_abs)!
        # 10! / (10+m)! * (10-m)! ... wait
        # (10-m)! / (10+m)!
        num = 1.0
        for i in range(10 - m_abs + 1, 11):
            num *= i
        den = 1.0
        for i in range(10 - m_abs + 1, 10 + m_abs + 1):
            den *= i
        # Actually simpler:
        # factorial(10-m_abs) / factorial(10+m_abs)
        import math
        val = math.factorial(10 - m_abs) / math.factorial(10 + m_abs)
        return np.sqrt((2 * L + 1) / (4 * np.pi) * val)
    
    # m = 0
    m = 0
    N0 = norm_const(0)
    P_10_0 = lpmv(0, 10, cos_theta) # lpmv includes Condon-Shortley phase (-1)^m
    Y[:, 10] = N0 * P_10_0.flatten() # Index 10 corresponds to m=0 (since m = i-10)
    
    # m > 0
    for m in range(1, L + 1):
        N_m = norm_const(m)
        P_10_m = lpmv(m, 10, cos_theta)
        # Y_m = sqrt(2) N_m P^m cos(m phi)
        val_cos = np.sqrt(2) * N_m * P_10_m * np.cos(m * phi_grid)
        # Y_-m = sqrt(2) N_m P^m sin(m phi)
        val_sin = np.sqrt(2) * N_m * P_10_m * np.sin(m * phi_grid)
        
        idx_pos = 10 + m
        idx_neg = 10 - m
        
        Y[:, idx_pos] = val_cos.flatten()
        Y[:, idx_neg] = val_sin.flatten()
        
    # 3. Define Objective Function and Gradient
    # We minimize R^4 = S4 / S2^2
    # S2 = sum W f^2
    # S4 = sum W f^4
    
    def objective_and_grad(c):
        f = Y @ c
        f2 = f * f
        f4 = f2 * f2
        
        S2 = np.dot(W_flat, f2)
        S4 = np.dot(W_flat, f4)
        
        # Avoid division by zero
        if S2 < 1e-12:
            return 1e10, np.zeros_like(c)
            
        val = S4 / (S2 * S2)
        
        # Gradient
        # dS2/dc = 2 * Y^T (W * f)
        # dS4/dc = 4 * Y^T (W * f^3)
        
        Wf = W_flat * f
        Wf3 = W_flat * f * f2
        
        grad_S2 = 2.0 * (Y.T @ Wf)
        grad_S4 = 4.0 * (Y.T @ Wf3)
        
        # d/dc (S4 / S2^2) = (S2^2 * dS4 - S4 * 2*S2 * dS2) / S2^4
        # = (dS4 * S2 - 2 * S4 * dS2) / S2^3
        # Wait:
        # Let g = S4 * S2^-2
        # dg = dS4 * S2^-2 + S4 * (-2) * S2^-3 * dS2
        # = (1/S2^2) * dS4 - (2*S4/S2^3) * dS2
        # = (1/S2^3) * (S2 * dS4 - 2 * S4 * dS2)
        
        grad = (S2 * grad_S4 - 2.0 * S4 * grad_S2) / (S2 ** 3)
        
        return val, grad

    # 4. Optimization
    best_val = np.inf
    best_c = np.ones(n_m)
    
    # Random starts
    np.random.seed(42)
    for i in range(20):
        # Random unit vector
        c0 = np.random.randn(n_m)
        c0 /= np.linalg.norm(c0)
        
        res = minimize(
            objective_and_grad, 
            c0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Also try some structured starts
    # e.g. only m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    res = minimize(objective_and_grad, c0, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_c = res.x
        
    # e.g. only m=10
    c0 = np.zeros(n_m)
    c0[20] = 1.0
    res = minimize(objective_and_grad, c0, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_c = res.x

    # Normalize the result to ensure it's within valid bounds
    # The ratio is scale invariant, so we can pick any scale.
    # Let's normalize so that max |c| = 1.
    max_c = np.max(np.abs(best_c))
    if max_c > 0:
        best_c = best_c / max_c
    else:
        best_c = np.ones(n_m) / np.sqrt(n_m)
        
    # Check validity
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_m)
        
    return best_c

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required block structure.
# EVOLVE-BLOCK-END
