import numpy as np
import scipy.optimize as opt
import time

# Hermite polynomials (Physicists')
def H0(x): return np.ones_like(x)
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 3840*x**6 + 13440*x**4 - 13440*x**2 + 1680
def H12(x):
    # 4096x^12 - 122880x^10 + 1290240x^8 - 6635520x^6 + 14515200x^4 - 14515200x^2 + 4790016
    # Coeffs: 2^12=4096.
    # H12(0) = 665280.
    return 4096*x**12 - 122880*x**10 + 1290240*x**8 - 6635520*x**6 + 14515200*x**4 - 14515200*x**2 + 4790016

# Pre-compute grid
X = np.linspace(-15, 15, 2000)
DX = X[1] - X[0]
GAUSS = np.exp(-X**2 / 2.0)

# Pre-compute polynomial parts
H0_G = H0(X) * GAUSS
H4_G = H4(X) * GAUSS
H8_G = H8(X) * GAUSS
H12_G = H12(X) * GAUSS

# Values at 0 for constraint
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

def compute_d(a, b, c):
    return -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0

def objective(params):
    a, b, c = params
    d = compute_d(a, b, c)
    g = a * H0_G + b * H4_G + c * H8_G + d * H12_G
    norm2_sq = np.sum(g**2) * DX
    norm4_sq = np.sum(g**4) * DX
    if norm2_sq <= 0:
        return 1e10
    return norm4_sq / norm2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin
    
    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Initial evaluation
    best_score = objective(x0)
    best_params = x0
    
    # Optimization
    try:
        # Use L-BFGS-B for speed and robustness
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if time.time() < deadline:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        
        # Refine with Nelder-Mead if time permits
        if time.time() < deadline - 20.0:
            res2 = opt.minimize(objective, best_params, method='Nelder-Mead', 
                                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x

    except Exception:
        pass

    # Return the coefficients a, b, c
    return np.array(best_params)
