import numpy as np
from scipy.optimize import minimize
import scipy.special
import time

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio ||f||_4 / ||f||_2.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights for cos(theta)
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    
    # Equally spaced phi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Flatten the grid
    cos_theta = np.cos(Theta).flatten()
    phi_flat = Phi.flatten()
    
    # Weights for integration
    w_phi = 2 * np.pi / n_phi
    weights = np.outer(w, np.ones(n_phi)).flatten() * w_phi
    
    # Precompute the basis functions Y_m^10
    basis = np.zeros((2 * k + 1, len(cos_theta)))
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        
        # Calculate factorial ratio (10-abs_m)! / (10+abs_m)!
        num = 1.0
        den = 1.0
        for i in range(10 - abs_m + 1, 10 + abs_m + 1):
            den *= i
        ratio = num / den
        
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
        
        # Associated Legendre polynomial with Condon-Shortley phase
        P = scipy.special.lpmv(abs_m, 10, cos_theta)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * phi_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi_flat)
            
        basis[m + 10] = Y
        
    # Objective function and gradient
    def objective_and_grad(c):
        f = basis @ c
        norm2_sq = np.sum(weights * f**2)
        norm4_sq = np.sum(weights * f**4)
        
        if norm2_sq < 1e-10:
            return 1e10, np.zeros_like(c)
            
        ratio_sq = norm4_sq / norm2_sq
        
        w_f3 = weights * f**3
        w_f = weights * f
        
        grad_A = 4.0 * (basis.T @ w_f3)
        grad_B = 2.0 * (basis.T @ w_f)
        
        grad_J = (norm2_sq * grad_A - norm4_sq * grad_B) / (norm2_sq**2)
        
        return ratio_sq, grad_J

    # Optimization
    best_c = np.ones(2 * k + 1)
    best_val = objective_and_grad(best_c)[0]
    
    start_time = time.time()
    timeout = 590.0
    
    # Try random initializations
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        c0 = np.random.randn(2 * k + 1)
        c0 = c0 / np.linalg.norm(c0)
        
        res = minimize(objective_and_grad, c0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            
    # Ensure validity
    max_c = np.max(np.abs(best_c))
    if max_c == 0:
        return np.ones(2 * k + 1)
        
    if max_c < 1e-3 or max_c > 1e5:
        best_c = best_c / max_c
        
    return best_c
