import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import time

# Hermite polynomials (Physicists')
def H0(x):
    return np.ones_like(x)

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 56700*x**4 - 75600*x**2 + 10395

# Values at 0
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

# Norms squared for normalization: <H_n e^{-x^2/2}, H_n e^{-x^2/2}> = sqrt(pi) * 2^n * n!
norm_sq_0 = np.sqrt(np.pi) * 1.0
norm_sq_4 = np.sqrt(np.pi) * 16 * 24.0 # 2^4 * 4! = 16 * 24 = 384
norm_sq_8 = np.sqrt(np.pi) * 256 * 40320.0 # 2^8 * 8!
norm_sq_12 = np.sqrt(np.pi) * 4096 * 479001600.0 # 2^12 * 12!

def calculate_c12(c0, c4, c8):
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    return (c0*H0_0 + c4*H4_0 + c8*H8_0) / (-H12_0)

def objective(params):
    c0, c4, c8 = params
    c12 = calculate_c12(c0, c4, c8)
    
    # Norm squared
    norm_sq = c0**2 * norm_sq_0 + c4**2 * norm_sq_4 + c8**2 * norm_sq_8 + c12**2 * norm_sq_12
    
    if norm_sq <= 0:
        return 1e10
        
    # Normalize coefficients so ||g||_2 = 1
    scale = 1.0 / np.sqrt(norm_sq)
    c0_n = c0 * scale
    c4_n = c4 * scale
    c8_n = c8 * scale
    c12_n = c12 * scale
    
    # Define g(x)
    def g(x):
        return (c0_n * H0(x) + c4_n * H4(x) + c8_n * H8(x) + c12_n * H12(x)) * np.exp(-x**2 / 2.0)
    
    # Define hat_g(xi)
    # F[H_n e^{-x^2/2}] = i^n H_n(xi) e^{-xi^2/2}
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = -1
    def hat_g(xi):
        return (c0_n * H0(xi) + c4_n * H4(xi) + c8_n * H8(xi) - c12_n * H12(xi)) * np.exp(-xi**2 / 2.0)
    
    # Integrate |g| and |hat_g|
    # Use quad for accuracy, with a timeout guard if possible, but quad is usually fast enough for simple functions
    # To be safe against slow quad, we can use a fixed grid if quad is too slow, but let's try quad first.
    # We restrict integration range to [-10, 10] as tails are negligible.
    
    try:
        I1, _ = quad(lambda x: abs(g(x)), -10, 10, limit=50)
        I2, _ = quad(lambda xi: abs(hat_g(xi)), -10, 10, limit=50)
    except Exception:
        return 1e10
        
    return I1 * I2

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, keep margin
    
    # Initial best
    best_val = 1e10
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate seed
    seed_val = objective(best_params)
    if seed_val < best_val:
        best_val = seed_val
        
    # Differential Evolution
    # Bounds: The scale doesn't matter due to normalization, but the ratios do.
    # We can bound the coefficients to a reasonable range, e.g., [-10, 10].
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6,
            update_interval=5,
            callback=lambda xk, convergence: (time.time() > deadline)
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_params
        
    # Optional: Local polish with L-BFGS-B or Nelder-Mead
    try:
        from scipy.optimize import minimize
        res2 = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
        )
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x
    except Exception:
        pass
        
    return best_params
