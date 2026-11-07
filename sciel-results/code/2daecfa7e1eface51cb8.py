import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 50

def compute_value(h):
    h = np.asarray(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return -np.inf
    S = np.sum(h)
    Q = np.sum(h**2)
    if S == 0 or Q == 0:
        return -np.inf
    g = np.convolve(h, h, mode='full')
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:])
    A = 2 * g_sq_sum + cross_sum
    return A / (3 * S**2 * Q)

def construct():
    # Grid points for x in [-1, 1]
    x = np.linspace(-1, 1, N)
    
    best_val = -np.inf
    best_h = np.ones(N)
    
    # Define parametric families
    # 1. Gaussian: exp(-a * x^2)
    def make_gaussian(a):
        return np.exp(-a * x**2)
        
    # 2. Power-law (on [-1,1]): (1 - x^2)^a
    def make_power(a):
        return np.power(np.maximum(0, 1 - x**2), a)
        
    # 3. Lorentzian: 1 / (1 + a * x^2)
    def make_lorentzian(a):
        return 1 / (1 + a * x**2)
        
    # 4. Sech: sech(a * x)
    def make_sech(a):
        return 1 / np.cosh(a * x)

    families = [
        (make_gaussian, [0.1, 10.0], 1.0),
        (make_power, [0.1, 5.0], 1.0),
        (make_lorentzian, [0.1, 10.0], 1.0),
        (make_sech, [0.1, 10.0], 1.0),
    ]
    
    for make_func, bounds, init in families:
        # Optimize the parameter 'a'
        # We want to maximize compute_value, so minimize negative value
        def objective(a):
            h = make_func(a)
            # Ensure non-negative and finite
            if np.any(h < 0) or not np.all(np.isfinite(h)):
                return 0
            return -compute_value(h)
        
        res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-8})
        a_opt = res.x
        h_opt = make_func(a_opt)
        val_opt = -res.fun
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
            
    # Also try a 2-parameter Gaussian: exp(-a * x^2) * (1 + b * x^2) ? 
    # Or just stick to 1-parameter for robustness and speed.
    # Let's add a "windowed" shape: h(x) = exp(-a*x^2) is already good.
    # What about h(x) = 1 - |x|^a ?
    def make_abs_power(a):
        return np.power(1 - np.abs(x), a)
        
    def objective2(a):
        h = make_abs_power(a)
        if np.any(h < 0) or not np.all(np.isfinite(h)):
            return 0
        return -compute_value(h)
        
    res2 = minimize_scalar(objective2, bounds=[0.1, 5.0], method='bounded', options={'xatol': 1e-8})
    a_opt2 = res2.x
    h_opt2 = make_abs_power(a_opt2)
    val_opt2 = -res2.fun
    
    if val_opt2 > best_val:
        best_val = val_opt2
        best_h = h_opt2

    # One more: Sinc-like but non-negative. 
    # h(x) = |sinc(x)| is not smooth. 
    # Let's try a "flat-top" Gaussian: 
    # h(x) = 1 if |x| < w, else exp(-a*(|x|-w))
    # This might be overkill. The simple parametric families usually capture the global optimum for such symmetric convex-like problems.
    
    # Final check: ensure non-negative
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) == 0:
        return list(np.ones(N))
        
    return list(best_h)
