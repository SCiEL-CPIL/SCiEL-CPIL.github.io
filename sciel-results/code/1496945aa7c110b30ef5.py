# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
from scipy.optimize import NonlinearConstraint

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
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(N_theta)
    
    u = x
    theta = np.arccos(u)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Create meshgrid
    U, P = np.meshgrid(u, phi, indexing='ij')
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Weights
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
    
    # Constraint: ||c||^2 = 1
    def norm_sq(c):
        return np.dot(c, c)
    
    def norm_sq_jac(c):
        return 2.0 * c
    
    # 4. Optimization
    best_c = None
    best_val = np.inf
    
    # Initial guesses
    inits = []
    # 1. All ones
    c0 = np.ones(2 * k + 1)
    c0 /= np.linalg.norm(c0)
    inits.append(c0)
    
    # 2. Only m=0
    c1 = np.zeros(2 * k + 1)
    c1[k] = 1.0
    inits.append(c1)
    
    # 3. Only m=k
    c2 = np.zeros(2 * k + 1)
    c2[-1] = 1.0
    inits.append(c2)
    
    # 4. Only m=-k
    c3 = np.zeros(2 * k + 1)
    c3[0] = 1.0
    inits.append(c3)
    
    # 5. Random unit vectors
    rng = np.random.default_rng(42)
    for _ in range(20):
        cr = rng.standard_normal(2 * k + 1)
        cr /= np.linalg.norm(cr)
        inits.append(cr)
        
    # 6. Symmetric even m
    c4 = np.zeros(2 * k + 1)
    for m in range(-k, k + 1, 2):
        c4[m + k] = 1.0
    c4 /= np.linalg.norm(c4)
    inits.append(c4)
    
    # 7. Symmetric odd m
    c5 = np.zeros(2 * k + 1)
    for m in range(-k, k + 1, 2):
        if m % 2 != 0:
            c5[m + k] = 1.0
    if np.linalg.norm(c5) > 0:
        c5 /= np.linalg.norm(c5)
        inits.append(c5)
        
    # 8. Zonal-like (cos(theta) powers)
    # This is basically m=0, already covered.
    
    # 9. Try to minimize variance (maximize flatness)
    # Start with coefficients proportional to 1/sqrt(N_m^2)?
    # N_m decreases as |m| increases.
    c9 = np.zeros(2 * k + 1)
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        log_num = np.log(2 * k + 1) - np.log(4 * np.pi) + np.log(factorial(k - abs_m)) - np.log(factorial(k + abs_m))
        N_m = np.sqrt(np.exp(log_num))
        c9[idx] = 1.0 / N_m if N_m > 0 else 1.0
    c9 /= np.linalg.norm(c9)
    inits.append(c9)

    # Use Trust-Constr which utilizes the Hessian
    for c_init in inits:
        try:
            res = minimize(
                objective, 
                c_init, 
                method='trust-constr',
                jac=gradient,
                hess=hessian,
                constraints=[NonlinearConstraint(norm_sq, 1.0, 1.0, jac=norm_sq_jac)],
                options={'maxiter': 1000, 'gtol': 1e-9, 'xtol': 1e-12, 'verbose': 0}
            )
            
            if res.success:
                val = res.fun
                # Ensure validity
                c_res = res.x
                if not np.any(np.isnan(c_res)) and not np.any(np.isinf(c_res)):
                    if val < best_val:
                        best_val = val
                        best_c = c_res
        except Exception:
            pass
            
    if best_c is None:
        best_c = np.ones(2 * k + 1)
        
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c

# EVOLVE-BLOCK-END
