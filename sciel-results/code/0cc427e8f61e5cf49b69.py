import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    k = 10
    # 1. Setup Quadrature
    # 70-point Gauss-Legendre in cos(theta)
    # We need nodes and weights for GL(70)
    # scipy.special.roots_legendre
    from scipy.special import roots_legendre
    x, w = roots_legendre(70)
    # x is in [-1, 1], w are weights for integral over [-1, 1]
    # Integral over sphere: \int_{-1}^1 du \int_0^{2\pi} d\phi
    # Total mass 4*pi.
    # GL weights sum to 2.
    # We need to map to sphere measure.
    # dOmega = dphi du.
    # So weights for u are w_i, for phi are 2*pi/N_phi.
    
    N_phi = 130
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = 2 * np.pi / N_phi
    
    # Grid
    theta = np.arccos(x)
    # Meshgrid
    U, PHI = np.meshgrid(x, phi, indexing='ij')
    # Weights
    W = np.outer(w, np.ones(N_phi)) * dphi
    
    # 2. Compute Spherical Harmonics Y_m on the grid
    # Y_0 = N_0 P^0
    # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    # P^m = P_10^m(cos theta) with Condon-Shortley phase.
    # scipy.special.lpmv(m, 10, cos_theta) includes Condon-Shortley phase (-1)^m.
    
    m_vals = np.arange(-k, k + 1)
    Y = np.zeros((2 * k + 1, len(x), N_phi))
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        # Normalization N_m
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Note: (10-m)! for m=10 is 0! = 1.
        # Factorial of non-integers? No, m is int.
        # Use gammaln for stability if needed, but 10 is small.
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * float(factorial(10 - abs_m)) / float(factorial(10 + abs_m)))
        
        P_m = lpmv(abs_m, 10, U) # U is cos(theta)
        
        if m == 0:
            Y[idx] = N_m * P_m
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N_m * P_m * np.cos(m * PHI)
        else:
            Y[idx] = np.sqrt(2.0) * N_m * P_m * np.sin(abs_m * PHI)
            
    # Flatten grid for easier vector ops
    Y_flat = Y.reshape(2 * k + 1, -1)
    W_flat = W.flatten()
    
    # 3. Define Objective Function
    # Minimize ||f||_4 / ||f||_2
    # Equivalent to minimizing (sum |f|^4) / (sum |f|^2)^2
    # Let S4 = sum W_flat * |f|^4
    # Let S2 = sum W_flat * |f|^2
    # J = S4 / S2^2
    
    def objective(c):
        f = Y_flat @ c
        abs_f_sq = f * f
        S2 = np.dot(W_flat, abs_f_sq)
        if S2 < 1e-15:
            return 1e15
        S4 = np.dot(W_flat, abs_f_sq * abs_f_sq)
        return S4 / (S2 * S2)
        
    # 4. Multi-start Optimization
    best_c = np.ones(2 * k + 1)
    best_val = objective(best_c)
    
    start_times = time.time()
    time_limit = 500.0 # Safety margin
    
    # Initial candidates
    candidates = [
        np.ones(2 * k + 1),
        np.random.randn(2 * k + 1),
    ]
    # Add single-term starts
    for i in range(2 * k + 1):
        v = np.zeros(2 * k + 1)
        v[i] = 1.0
        candidates.append(v)
        
    # Add some random
    for _ in range(10):
        candidates.append(np.random.randn(2 * k + 1))
        
    # Normalize candidates
    candidates = [c / np.linalg.norm(c) for c in candidates]
    
    for c0 in candidates:
        if time.time() - start_times > time_limit:
            break
            
        # Use Powell, it's derivative-free and robust for this size
        # Or L-BFGS-B with numerical gradient?
        # Powell is safer for non-smoothness (though this is smooth).
        res = minimize(objective, c0, method='Powell', options={'maxiter': 1000, 'disp': False})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    # Normalize final result
    best_c = best_c / np.linalg.norm(best_c)
    
    # Validation
    if not np.all(np.isfinite(best_c)):
        return np.ones(2 * k + 1)
    if np.max(np.abs(best_c)) < 1e-3 or np.max(np.abs(best_c)) > 1e5:
        # Scale if necessary, though normalization keeps it ~1/sqrt(21) ~ 0.2
        # If it's too small, something went wrong.
        pass
        
    return best_c
