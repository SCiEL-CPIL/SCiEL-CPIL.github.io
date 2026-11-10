import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    k = 10
    n_coeffs = 2 * k + 1
    
    n_theta = 70
    n_phi = 130
    
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(cos_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    theta_flat = Theta.flatten()
    phi_flat = Phi.flatten()
    
    w_total = np.outer(w_theta, np.full(n_phi, w_phi)).flatten()
    
    Y_vals = np.zeros((n_theta * n_phi, n_coeffs))
    
    for idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        P_lm = lpmv(abs_m, k, cos_theta)
        
        factorial_ratio = np.math.factorial(k - abs_m) / np.math.factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * factorial_ratio)
        
        if m == 0:
            Y_vals[:, idx] = N_m * P_lm
        elif m > 0:
            Y_vals[:, idx] = np.sqrt(2) * N_m * P_lm * np.cos(m * phi_flat)
        else:
            Y_vals[:, idx] = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * phi_flat)
            
    # Verify orthonormality numerically?
    # G = Y.T @ (w_total[:, None] * Y)
    # Should be close to Identity.
    
    def objective_and_grad(c):
        f = Y_vals @ c
        f2 = f**2
        f4 = f2**2
        I2 = np.sum(w_total * f2)
        I4 = np.sum(w_total * f4)
        
        if I2 < 1e-15:
            return 1e15, np.zeros_like(c)
            
        val = I4 / (I2**2)
        
        # Gradient of I4 / I2^2
        # d/dc (I4 I2^-2) = I2^-2 dI4/dc - 2 I4 I2^-3 dI2/dc
        # dI2/dc = 2 Y.T (w_total * f)
        # dI4/dc = 4 Y.T (w_total * f^3)
        
        dI2 = 2 * (Y_vals.T @ (w_total * f))
        dI4 = 4 * (Y_vals.T @ (w_total * f**3))
        
        grad = (I2**-2) * dI4 - 2 * (I4 * I2**-3) * dI2
        grad = grad / (I2**2) # Wait, let's re-derive carefully.
        
        # Let F = I4 / I2^2
        # grad F = (1/I2^2) grad I4 - I4 * 2 * I2^-3 * grad I2
        # grad I4 = 4 Y^T (w f^3)
        # grad I2 = 2 Y^T (w f)
        
        grad = (dI4 / (I2**2)) - (2 * I4 * dI2) / (I2**3)
        
        return val, grad
    
    c_init = np.ones(n_coeffs)
    
    best_c = c_init
    best_val, _ = objective_and_grad(c_init)
    
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > deadline - 10:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(lambda x: objective_and_grad(x)[0], c_init, method='BFGS', 
                       jac=lambda x: objective_and_grad(x)[1],
                       options={'maxiter': 200, 'gtol': 1e-8},
                       callback=callback)
        if res.fun < best_val:
            best_c = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Try a few random restarts if time permits
    if time.time() - start_time < deadline - 30:
        rng = np.random.default_rng(123)
        for i in range(5):
            if time.time() - start_time > deadline - 15:
                break
            c_rand = rng.normal(size=n_coeffs)
            try:
                res_r = minimize(lambda x: objective_and_grad(x)[0], c_rand, method='BFGS', 
                               jac=lambda x: objective_and_grad(x)[1],
                               options={'maxiter': 100, 'gtol': 1e-8},
                               callback=callback)
                if res_r.fun < best_val:
                    best_c = res_r.x
                    best_val = res_r.fun
            except Exception:
                pass

    if not np.all(np.isfinite(best_c)):
        return np.ones(n_coeffs)
        
    norm_c = np.linalg.norm(best_c)
    if norm_c > 0:
        best_c = best_c / norm_c
    else:
        best_c = np.ones(n_coeffs)
        
    return best_c
