import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import math

def H0(x):
    return 1

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 55440*x**2 + 10395

def construct():
    # Pre-compute Gauss-Hermite nodes and weights
    # We need high accuracy for 4th power integrals (degree 48 polynomial)
    # N=100 is sufficient for exactness of polynomials up to degree 199
    x_gh, w_gh = roots_hermite(100)
    
    # We need to integrate P(x)^4 e^{-2x^2}
    # Let y = sqrt(2) x => x = y / sqrt(2)
    # dx = dy / sqrt(2)
    # Integral = (1/sqrt(2)) * Integral P(y/sqrt(2))^4 e^{-y^2} dy
    # We can evaluate P at x_gh / sqrt(2) and weight by w_gh / sqrt(2)
    
    x_eval = x_gh / np.sqrt(2.0)
    w_eval = w_gh / np.sqrt(2.0)
    
    # Pre-compute Hermite values at x_eval for efficiency
    H0_vals = np.ones_like(x_eval)
    H4_vals = x_eval**4 - 6*x_eval**2 + 3
    H8_vals = x_eval**8 - 28*x_eval**6 + 210*x_eval**4 - 420*x_eval**2 + 105
    H12_vals = x_eval**12 - 60*x_eval**10 + 1260*x_eval**8 - 12600*x_eval**6 + 45360*x_eval**4 - 55440*x_eval**2 + 10395
    
    # We also need integrals for P(x)^2 e^{-x^2}
    # This is standard GH quadrature with x_gh and w_gh
    # But we can also use x_eval and w_eval? No, the weight is different.
    # Let's just compute I2 separately or reuse.
    # I2 = Integral P(x)^2 e^{-x^2} dx = sum w_gh * P(x_gh)^2
    # Let's pre-compute H values at x_gh as well
    H0_vals_2 = np.ones_like(x_gh)
    H4_vals_2 = x_gh**4 - 6*x_gh**2 + 3
    H8_vals_2 = x_gh**8 - 28*x_gh**6 + 210*x_gh**4 - 420*x_gh**2 + 105
    H12_vals_2 = x_gh**12 - 60*x_gh**10 + 1260*x_gh**8 - 12600*x_gh**6 + 45360*x_gh**4 - 55440*x_gh**2 + 10395

    def objective(params):
        a4, a8 = params
        a0 = 1.0
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        
        # Calculate P(x) and Q(x) at x_eval for L4 integrals
        # P = a0 H0 + a4 H4 + a8 H8 + a12 H12
        P_vals = a0 * H0_vals + a4 * H4_vals + a8 * H8_vals + a12 * H12_vals
        # Q = a0 H0 - a4 H4 + a8 H8 - a12 H12
        Q_vals = a0 * H0_vals - a4 * H4_vals + a8 * H8_vals - a12 * H12_vals
        
        I4_g = np.sum(w_eval * (P_vals**4))
        I4_ghat = np.sum(w_eval * (Q_vals**4))
        
        # Calculate I2 for L2 norm
        # P_2 = a0 H0 + a4 H4 + a8 H8 + a12 H12 at x_gh
        P_vals_2 = a0 * H0_vals_2 + a4 * H4_vals_2 + a8 * H8_vals_2 + a12 * H12_vals_2
        I2_g = np.sum(w_gh * (P_vals_2**2))
        
        # The ratio to minimize is:
        # (I4_g * I4_ghat)^(1/4) / I2_g
        # To avoid roots, minimize (I4_g * I4_ghat) / I2_g^4
        # This is monotonic and positive.
        
        if I2_g <= 0:
            return 1e100
            
        val = (I4_g * I4_ghat) / (I2_g**4)
        return val

    # Initial guesses
    # Seed: a0=-1.13, a4=0.04, a8=1/3240
    # Scaled to a0=1:
    seed_a0 = -1.13
    seed_a4 = 0.04
    seed_a8 = 1.0 / 3240.0
    
    init_a4 = seed_a4 / seed_a0
    init_a8 = seed_a8 / seed_a0
    
    # Try a few different starting points to ensure we find the global minimum
    starts = [
        [init_a4, init_a8],
        [0.0, 0.0],
        [0.1, 0.0],
        [-0.1, 0.0],
        [0.0, 0.1],
        [0.0, -0.1],
        [init_a4 * 2, init_a8 * 2],
        [init_a4 * 0.5, init_a8 * 0.5]
    ]
    
    best_val = float('inf')
    best_x = None
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to seed if optimization failed
        return np.array([1.0, init_a4, init_a8])
        
    best_a4, best_a8 = best_x
    best_a0 = 1.0
    
    return np.array([best_a0, best_a4, best_a8])
