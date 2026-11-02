import numpy as np
import time
from scipy.optimize import linprog, minimize
from scipy.special import factorial

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200  # Safety margin

def _time_left():
    return _DEADLINE - time.time()

# Hermite polynomial coefficients for H0, H4, H8, H12
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3  (Wait, standard physicist's Hermite polynomials? 
# The prompt's current program uses:
# H4 = [12, 0, -48, 0, 16] -> 16x^4 - 48x^2 + 12 = 4(4x^4 - 12x^2 + 3)
# Standard H4 (physicist) is x^4 - 6x^2 + 3.
# The current program uses a scaled version? 
# Let's check H8 in current program: [10080, ..., 256] -> 256x^8 - ...
# Standard H8 is x^8 - 28x^6 + 210x^4 - 420x^2 + 105.
# 256 * 105 = 26880. Current has 10080?
# 10080 / 105 = 96.
# 16/1 = 16.
# It seems the current program uses specific coefficients. 
# Inspiration 3 uses:
# H4 = 16x^4 - 48x^2 + 12 (matches current H4/4? No, current H4 is 16x^4... wait.
# Current H4 array: [12.0, 0.0, -48.0, 0.0, 16.0] -> 12 - 48x^2 + 16x^4.
# This is 4 * (3 - 12x^2 + 4x^4).
# Standard H4 is 1 - 6x^2 + x^4? No, H4(x) = x^4 - 6x^2 + 3.
# 4*H4(x) = 4x^4 - 24x^2 + 12.
# Current is 16x^4 - 48x^2 + 12.
# This is NOT a standard scaling.
# Let's look at H8 in current: 10080 - 34560x^2 + 23040x^4 - 4608x^6 + 256x^8.
# Divide by 256: 39.375 - 135x^2 + 90x^4 - 18x^6 + x^8.
# Standard H8: 105 - 420x^2 + 210x^4 - 28x^6 + x^8.
# 105/39.375 = 2.666.
# It seems the current program defines its own polynomials. 
# HOWEVER, the task says "Hermite-polynomial". Usually implies standard ones.
# But the seed coefficients [-1.13, 0.04, 0.000308] are tuned for the SPECIFIC polynomials in the current program's context?
# If I change the polynomials, the seed coefficients will not make sense.
# I MUST use the SAME polynomials as the current program to be comparable to the seed.
# Let's extract the polynomials from the current program's arrays.
# H0 = [1] -> 1
# H4 = [12, 0, -48, 0, 16] -> 12 - 48x^2 + 16x^4
# H8 = [10080, 0, -34560, 0, 23040, 0, -4608, 0, 256] -> 10080 - 34560x^2 + 23040x^4 - 4608x^6 + 256x^8
# H12 = [10321920, ..., 4096] -> 10321920 - ... + 4096x^12

# Let's verify H12_0 in current program: 10321920.
# H8_0: 10080.
# H4_0: 12.
# H0_0: 1.

# The constraint g(0)=0:
# c0*1 + c4*12 + c8*10080 + c12*10321920 = 0.
# c12 = -(c0 + 12c4 + 10080c8) / 10321920.

# I will use these exact polynomials.
# To evaluate g(x) = c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x),
# I can use np.polyval if I store coefficients in descending order, or just evaluate directly.
# The arrays in current program are in ASCENDING order of power (constant term first).
# H0 = [1]
# H4 = [12, 0, -48, 0, 16]
# H8 = [10080, 0, -34560, 0, 23040, 0, -4608, 0, 256]
# H12 = [10321920, 0, -34406400, 0, 34406400, 0, -12902400, 0, 2090112, 0, -147456, 0, 4096]

# For Gauss-Hermite quadrature, we need to evaluate these polynomials at x_gh.
# I will define functions to evaluate them.

def _eval_H0(x):
    return np.ones_like(x)

def _eval_H4(x):
    return 12.0 - 48.0*x**2 + 16.0*x**4

def _eval_H8(x):
    x2 = x**2
    return 10080.0 - 34560.0*x2 + 23040.0*x2**2 - 4608.0*x2**3 + 256.0*x2**4

def _eval_H12(x):
    x2 = x**2
    return (10321920.0 - 34406400.0*x2 + 34406400.0*x2**2 
            - 12902400.0*x2**3 + 2090112.0*x2**4 
            - 147456.0*x2**5 + 4096.0*x2**6)

def _compute_g(x, c0, c4, c8):
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 10080.0) / 10321920.0
    return c0 * _eval_H0(x) + c4 * _eval_H4(x) + c8 * _eval_H8(x) + c12 * _eval_H12(x)

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = seed_result.copy()
    
    # We want to minimize A(f) = sup |x * g(x) * exp(-x^2/2)|
    # Using Gauss-Hermite Quadrature
    N = 200
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    # Precompute H values at quadrature points
    H0_vals = _eval_H0(x_gh)
    H4_vals = _eval_H4(x_gh)
    H8_vals = _eval_H8(x_gh)
    H12_vals = _eval_H12(x_gh)
    
    # g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 12c4 + 10080c8)/10321920
    # Let c = [c0, c4, c8].
    # g(x_i) = c0*(H0_i - H12_i/10321920) + c4*(H4_i - 12*H12_i/10321920) + c8*(H8_i - 10080*H12_i/10321920)
    
    H12_scaled = H12_vals / 10321920.0
    M0 = H0_vals - H12_scaled
    M4 = H4_vals - 12.0 * H12_scaled
    M8 = H8_vals - 10080.0 * H12_scaled
    
    # We want to minimize t such that |x_i * g(x_i) * exp(-x_i^2/2)| <= t
    # Let w_i = x_i * exp(-x_i^2/2)
    w_vals = x_gh * np.exp(-x_gh**2 / 2.0)
    
    # Constraints:
    # -t <= w_i * (c0*M0_i + c4*M4_i + c8*M8_i) <= t
    # -t - w_i*(c0*M0_i + c4*M4_i + c8*M8_i) <= 0
    # t - w_i*(c0*M0_i + c4*M4_i + c8*M8_i) >= 0  => -t + w_i*(...) <= 0
    
    # Variables: [c0, c4, c8, t]
    # Objective: minimize t -> c = [0, 0, 0, 1]
    
    c_obj = np.array([0.0, 0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    for i in range(N):
        wi = w_vals[i]
        # Constraint 1: wi*(c0*M0_i + c4*M4_i + c8*M8_i) - t <= 0
        # c0*(wi*M0_i) + c4*(wi*M4_i) + c8*(wi*M8_i) - t <= 0
        A_ub.append([wi*M0_vals[i], wi*M4_vals[i], wi*M8_vals[i], -1.0])
        b_ub.append(0.0)
        
        # Constraint 2: -(wi*(c0*M0_i + c4*M4_i + c8*M8_i)) - t <= 0
        # -c0*(wi*M0_i) - c4*(wi*M4_i) - c8*(wi*M8_i) - t <= 0
        A_ub.append([-wi*M0_vals[i], -wi*M4_vals[i], -wi*M8_vals[i], -1.0])
        b_ub.append(0.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for c0, c4, c8 to prevent numerical issues (large values)
    # Seed norm is approx 1.13. Let's allow range [-10, 10]
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (0.0, None)]
    
    if _time_left() > 10:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                best_result = res.x[:3]
        except Exception:
            pass
    
    # Polish with local optimization on the actual max norm (or high power)
    # Define objective: max |x g(x) exp(-x^2/2)|
    def obj_lp(c):
        g = c[0]*M0 + c[1]*M4 + c[2]*M8
        vals = w_vals * g
        return np.max(np.abs(vals))
    
    if _time_left() > 5:
        try:
            res = minimize(obj_lp, best_result, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < obj_lp(best_result):
                best_result = res.x
        except Exception:
            pass
            
    return best_result
