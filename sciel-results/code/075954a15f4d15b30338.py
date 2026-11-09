import numpy as np

def construct():
    # Fit coefficients for the ansatz:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # to the Linder-Vasquez approximation:
    # gamma_target = (5/7) * (Omega_m / u)^0.38
    
    Om0 = 0.3
    
    # Grid focused on "smooth" CPL cosmologies
    w0_vals = np.linspace(-1.0, 0.0, 40)
    wa_vals = np.linspace(-0.5, 0.5, 40)
    a_vals = np.linspace(0.1, 1.0, 80)
    
    W0, WA, A = np.meshgrid(w0_vals, wa_vals, a_vals, indexing='ij')
    W0 = W0.flatten()
    WA = WA.flatten()
    A = A.flatten()
    
    U = 1.0 + W0 + WA * (1.0 - A)
    
    # Avoid u <= 0 for the target calculation
    valid_u = U > 1e-6
    
    if not np.any(valid_u):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    W0_v = W0[valid_u]
    WA_v = WA[valid_u]
    A_v = A[valid_u]
    U_v = U[valid_u]
    
    denom = Om0 * A_v**3 + (1.0 - Om0) * A_v**(-3.0 * WA_v) * np.exp(-3.0 * (1.0 + W0_v) * A_v)
    Om_a = Om0 * A_v**3 / denom
    Q = 1.0 - Om_a
    
    ratio = Om_a / U_v
    # Ensure ratio is positive
    valid_ratio = ratio > 1e-6
    if not np.any(valid_ratio):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    W0_v = W0_v[valid_ratio]
    WA_v = WA_v[valid_ratio]
    A_v = A_v[valid_ratio]
    U_v = U_v[valid_ratio]
    Q = Q[valid_ratio]
    ratio = ratio[valid_ratio]
    
    gamma_target = (5.0/7.0) * np.power(ratio, 0.38)
    
    # Filter to target range [0.1, 1.2] to keep the fit well-behaved
    mask = (gamma_target >= 0.1) & (gamma_target <= 1.2)
    
    if np.sum(mask) < 100:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    Q_m = Q[mask]
    U_m = U_v[mask]
    WA_m = WA_v[mask]
    A_m = A_v[mask]
    gamma_m = gamma_target[mask]
    
    X = np.column_stack([
        np.ones_like(Q_m),
        Q_m,
        Q_m**2,
        U_m,
        WA_m * A_m,
        U_m * Q_m
    ])
    
    coeffs, residuals, rank, s = np.linalg.lstsq(X, gamma_m, rcond=None)
    
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    return coeffs.tolist()
