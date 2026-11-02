import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid
x = np.linspace(-8, 8, 4001)
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

def objective(coeffs):
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    l1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform: FT of He_n(x)*exp(-x^2/2) is (i)^n * He_n(xi)*exp(-xi^2/2)
    # Using probabilists' convention: FT[f](xi) = integral f(x) exp(-i x xi) dx
    # For g(x) = P(x) exp(-x^2/2), FT[g](xi) = sqrt(2*pi) * P(i*xi) * exp(-xi^2/2)
    # where P(i*xi) = c0*He_0(i*xi) + c1*He_4(i*xi) + c2*He_8(i*xi) + c3*He_12(i*xi)
    # He_n(i*xi) = (i)^n * He_n(xi) for probabilists' Hermite polynomials
    
    xi = x  # symmetric grid
    P_at_i_xi = (c0*1.0 + c1*(xi**4 - 6*xi**2 + 3) + c2*(xi**8 - 28*xi**6 + 420*xi**4 - 840*xi**2 + 105) + c3*(xi**12 - 66*xi**10 + 2310*xi**8 - 39900*xi**6 + 332640*xi**4 - 798336*xi**2 + 10395))
    
    # Actually, He_n(i*xi) = (i)^n He_n(xi), so |He_n(i*xi)| = |He_n(xi)|
    # The modulus of FT[g](xi) = sqrt(2*pi) * |P(i*xi)| * exp(-xi^2/2)
    # But P(i*xi) = sum c_k * He_k(i*xi) = sum c_k * (i)^k * He_k(xi)
    # So |P(i*xi)| = |sum c_k * (i)^k * He_k(xi)|
    
    # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = 1
    # So all (i)^k = 1 for k = 0,4,8,12
    # Therefore P(i*xi) = c0*He_0(xi) + c1*He_4(xi) + c2*He_8(xi) + c3*He_12(xi)
    
    hat_g = np.sqrt(2*np.pi) * (c0*he0(xi) + c1*he4(xi) + c2*he8(xi) + c3*he12(xi)) * np.exp(-xi**2/2)
    
    min_abs_hat = np.min(np.abs(hat_g))
    if min_abs_hat < 1e-10:
        return 1e10
    return l1 / min_abs_hat

def construct():
    t0 = time.time()
    deadline = t0 + 170  # safety margin
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)
    
    # Local refinement from seed
    try:
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol':1e-9, 'fatol':1e-11})
        if time.time() < deadline and res.fun < best_val:
            best = res.x.copy()
            best_val = res.fun
    except Exception:
        pass
    
    # Multi-start with diverse initial points
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        # Try points in a wider range
        for _ in range(30):
            if time.time() > deadline - 30:
                break
            start = rng.uniform(-2, 2, size=3)
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol':1e-8, 'fatol':1e-10})
                if time.time() > deadline - 30:
                    break
                if res.fun < best_val:
                    best = res.x.copy()
                    best_val = res.fun
            except Exception:
                pass
    
    # Final local polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol':1e-10, 'fatol':1e-12})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    return np.array(best)
