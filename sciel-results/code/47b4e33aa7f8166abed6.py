# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Finds the real coefficients c for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_m c_m Y_m that minimize the ratio ||f||_4 / ||f||_2.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    k = 10
    
    # 1. Setup Quadrature Grid
    N_theta = 70
    N_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(N_theta)
    
    u = x
    theta = np.arccos(u)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    U, P = np.meshgrid(u, phi, indexing='ij')
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    dphi = 2 * np.pi / N_phi
    W_grid = np.outer(w, np.full(N_phi, dphi))
    W_flat = W_grid.flatten()
    
    # 2. Precompute Spherical Harmonics Y_m(θ, φ)
    m_vals = np.arange(-k, k + 1)
    Y_flat = np.zeros((N_theta * N_phi, 2 * k + 1))
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        P_val = lpmv(abs_m, k, U)
        
        log_num = np.log(2 * k + 1) - np.log(4 * np.pi) + np.log(factorial(k - abs_m)) - np.log(factorial(k + abs_m))
        N_m = np.sqrt(np.exp(log_num))
        
        if m == 0:
            Y_val = N_m * P_val
        elif m > 0:
            Y_val = np.sqrt(2) * N_m * P_val * np.cos(m * PHI)
        else:
            Y_val = np.sqrt(2) * N_m * P_val * np.sin(abs_m * PHI)
        
        Y_flat[:, idx] = Y_val.flatten()
    
    # 3. Define Objective, Gradient, and Hessian
    def objective(c):
        f = Y_flat @ c
        return np.sum(W_flat * (f ** 4))
    
    def gradient(c):
        f = Y_flat @ c
        return 4.0 * (Y_flat.T @ (W_flat * (f ** 3)))
    
    def hessian(c):
        f = Y_flat @ c
        w_f2 = W_flat * (f ** 2)
        Y_weighted = Y_flat * w_f2[:, None]
        return 12.0 * (Y_weighted.T @ Y_flat)
    
    # 4. Optimization using trust-constr to leverage Hessian
    best_c = None
    best_val = np.inf
    
    inits = []
    c0 = np.ones(2 * k + 1)
    c0 /= np.linalg.norm(c0)
    inits.append(c0)
    
    c1 = np.zeros(2 * k + 1)
    c1[k] = 1.0
    inits.append(c1)
    
    c2 = np.zeros(2 * k + 1)
    c2[-1] = 1.0
    inits.append(c2)
    
    c3 = np.zeros(2 * k + 1)
    c3[0] = 1.0
    inits.append(c3)
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        cr = rng.standard_normal(2 * k + 1)
        cr /= np.linalg.norm(cr)
        inits.append(cr)
        
    c4 = np.zeros(2 * k + 1)
    for m in range(-k, k + 1, 2):
        c4[m + k] = 1.0
    c4 /= np.linalg.norm(c4)
    inits.append(c4)
    
    c5 = np.zeros(2 * k + 1)
    for i, m in enumerate(m_vals):
        c5[i] = 1.0 if m % 2 == 0 else -1.0
    c5 /= np.linalg.norm(c5)
    inits.append(c5)
        
    for c_init in inits:
        try:
            res = minimize(
                objective, 
                c_init, 
                method='trust-constr',
                jac=gradient,
                hess=hessian,
                constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2 * c}],
                options={'maxiter': 1000, 'gtol': 1e-12, 'xtol': 1e-12}
            )
            
            if res.success or (hasattr(res, 'fun') and res.fun < best_val):
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
        except Exception:
            pass
            
    if best_c is None:
        best_c = np.ones(2 * k + 1)
        
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c

# EVOLVE-BLOCK-END
