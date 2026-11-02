import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid with higher resolution
x = np.linspace(-10, 10, 8001)
dx = x[1] - x[0]
gauss = np.exp(-x**2 / 2)

# Probabilists' Hermite polynomials He_0, He_4, He_8, He_12 evaluated on x
def he0(x): return np.ones_like(x)
def he4(x): return x**4 - 6*x**2 + 3
def he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

H0 = he0(x)
H4 = he4(x)
H8 = he8(x)
H12 = he12(x)

# Values at 0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def soft_min(arr, temperature=1e-3):
    """Smooth approximation of min using log-sum-exp trick"""
    neg_arr = -arr
    max_val = np.max(neg_arr)
    return -max_val - temperature * np.log(np.sum(np.exp((neg_arr - max_val) / temperature)))

def objective(coeffs, temperature=1e-3):
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    l1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform using the fact that (i)^k = 1 for k = 0,4,8,12
    hat_g = np.sqrt(2*np.pi) * (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    
    abs_hat = np.abs(hat_g)
    # Use soft-min for smoothness during optimization
    min_abs_hat = soft_min(abs_hat, temperature)
    if min_abs_hat < 1e-10:
        return 1e10
    return l1 / min_abs_hat

def objective_exact(coeffs):
    """Exact objective for final evaluation"""
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    l1 = np.sum(np.abs(g)) * dx
    
    hat_g = np.sqrt(2*np.pi) * (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    min_abs_hat = np.min(np.abs(hat_g))
    if min_abs_hat < 1e-10:
        return 1e10
    return l1 / min_abs_hat

def construct():
    t0 = time.time()
    deadline = t0 + 190  # safety margin before 210s kill
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective_exact(best)
    
    # Phase 1: Local refinement from seed with smooth objective
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol':1e-10, 'fatol':1e-12})
        if time.time() < deadline and res.fun < best_val:
            best = res.x.copy()
            best_val = objective_exact(best)
    except Exception:
        pass
    
    # Phase 2: Multi-start with diverse initial points
    if time.time() < deadline - 80:
        rng = np.random.default_rng(42)
        # Generate diverse starting points in multiple scales
        starts = []
        # Small perturbations around seed
        for _ in range(10):
            starts.append(seed + rng.normal(0, 0.1, size=3))
        # Medium range
        for _ in range(10):
            starts.append(rng.uniform(-1, 1, size=3))
        # Larger range
        for _ in range(10):
            starts.append(rng.uniform(-3, 3, size=3))
        # Structured points
        for scale in [0.01, 0.1, 1.0]:
            for i in range(3):
                pt = np.zeros(3)
                pt[i] = scale
                starts.append(pt)
                pt[i] = -scale
                starts.append(pt)
        
        for start in starts:
            if time.time() > deadline - 40:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol':1e-9, 'fatol':1e-11})
                if time.time() > deadline - 40:
                    break
                val = objective_exact(res.x)
                if val < best_val:
                    best = res.x.copy()
                    best_val = val
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with exact objective
    if time.time() < deadline - 15:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol':1e-11, 'fatol':1e-13})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 4: One more very tight polish if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(objective_exact, best, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol':1e-12, 'fatol':1e-14})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
