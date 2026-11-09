import numpy as np
import scipy.optimize as opt
import time

# Hermite polynomials (Physicists')
def H0(x): return np.ones_like(x)
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 3840*x**6 + 13440*x**4 - 13440*x**2 + 1680
def H12(x):
    return 4096*x**12 - 122880*x**10 + 1290240*x**8 - 6635520*x**6 + 14515200*x**4 - 14515200*x**2 + 4790016

# Pre-compute grid - use 1000 points for speed, sufficient accuracy
X = np.linspace(-15, 15, 1000)
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
    deadline = start_time + 190.0  # Safety margin before 210s kill
    
    # Seed values
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Initial evaluation
    best_score = objective(x0)
    best_params = x0.copy()
    
    # Define bounds for coefficients based on seed magnitude
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Phase 1: Global search with differential evolution (robust to local minima)
    try:
        if time.time() < deadline - 60.0:
            de_res = opt.differential_evolution(
                objective, 
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=False,
                update_interval=10
            )
            if time.time() < deadline - 30.0:
                if de_res.fun < best_score:
                    best_score = de_res.fun
                    best_params = de_res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B from best point found
    try:
        if time.time() < deadline - 20.0:
            lbfgs_res = opt.minimize(
                objective, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if lbfgs_res.fun < best_score:
                best_score = lbfgs_res.fun
                best_params = lbfgs_res.x.copy()
    except Exception:
        pass
    
    # Phase 3: Final polish with Nelder-Mead if time permits
    try:
        if time.time() < deadline - 5.0:
            nm_res = opt.minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000}
            )
            if nm_res.fun < best_score:
                best_score = nm_res.fun
                best_params = nm_res.x.copy()
    except Exception:
        pass
    
    # Return the coefficients a, b, c
    return np.array(best_params)
