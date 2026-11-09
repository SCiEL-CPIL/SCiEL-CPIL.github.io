import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize
from scipy.integrate import gauss_legendre

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0 # 10s safety margin
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    n_theta = 70
    n_phi = 130
    
    x, w = gauss_legendre(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Precompute P_10^m(x)
    fact = np.math.factorial
    N = np.zeros(k+1)
    for m in range(k+1):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * fact(k-m) / fact(k+m))
        
    P = np.zeros((n_theta, k+1))
    for m in range(k+1):
        P[:, m] = lpmv(m, k, x)
        
    Y_grid = np.zeros((n_theta, n_phi, n_coeffs))
    
    for i, m in enumerate(range(-k, k+1)):
        abs_m = abs(m)
        P_m = P[:, abs_m]
        
        if m == 0:
            val = N[0] * P_m
            Y_grid[:, :, i] = val[:, np.newaxis]
        elif m > 0:
            val = np.sqrt(2.0) * N[m] * P_m * np.cos(m * phi)[np.newaxis, :]
            Y_grid[:, :, i] = val
        else:
            val = np.sqrt(2.0) * N[abs_m] * P_m * np.sin(abs_m * phi)[np.newaxis, :]
            Y_grid[:, :, i] = val
            
    Y_flat = Y_grid.reshape(-1, n_coeffs)
    
    weights = w[:, np.newaxis] * (2.0 * np.pi / n_phi)
    weights_flat = weights.reshape(-1)
    
    def objective(c):
        f = Y_flat @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.dot(weights_flat, f2)
        norm4_4th = np.dot(weights_flat, f4)
        
        if norm2_sq <= 0 or norm4_4th <= 0:
            return 1e10
            
        return 0.5 * np.log(norm4_4th) - np.log(norm2_sq)
        
    def objective_grad(c):
        f = Y_flat @ c
        f2 = f * f
        
        norm2_sq = np.dot(weights_flat, f2)
        norm4_4th = np.dot(weights_flat, f2 * f2)
        
        if norm2_sq <= 0 or norm4_4th <= 0:
            return np.zeros_like(c)
            
        grad_N2 = 2.0 * (Y_flat.T @ (weights_flat * f))
        grad_N4 = 4.0 * (Y_flat.T @ (weights_flat * f**3))
        
        grad = 0.5 * (grad_N4 / norm4_4th) - (grad_N2 / norm2_sq)
        
        return grad
        
    best_val = np.inf
    best_c = np.ones(n_coeffs)
    
    # Dynamic restarts
    # Estimate time for first few
    n_restarts_done = 0
    max_restarts = 200
    
    for i in range(max_restarts):
        if time.time() - start_time > deadline:
            break
            
        c_init = np.random.randn(n_coeffs)
        c_init /= np.linalg.norm(c_init)
        
        res = minimize(objective, c_init, jac=objective_grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        n_restarts_done += 1
        
        if res.success:
            val = res.fun
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
                
        # If we are running fast, we can continue.
        # If slow, the deadline guard handles it.
        
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_coeffs)
        
    return best_c
