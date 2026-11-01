import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = 165.0  # 15s safety margin
    
    def H0(x):
        return 1.0
    
    def H4(x):
        return 16.0*x**4 - 48.0*x**2 + 12.0
    
    def H8(x):
        return 256.0*x**8 - 3072.0*x**6 + 11520.0*x**4 - 13440.0*x**2 + 1680.0
    
    def H12(x):
        return 4096.0*x**12 - 122880.0*x**10 + 1843200.0*x**8 - 12902400.0*x**6 + 44352000.0*x**4 - 71232000.0*x**2 + 10321600.0
    
    def P_val(x, a0, a4, a8):
        a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 10321600.0
        return a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    
    def integrand(x, a0, a4, a8):
        return abs(P_val(x, a0, a4, a8) * np.exp(-0.5*x*x))
    
    def objective(params):
        if time.time() - start_time > deadline - 5:
            return 1e10
        a0, a4, a8 = params
        try:
            val, _ = quad(integrand, -15.0, 15.0, args=(a0, a4, a8), limit=300, epsabs=1e-12, epsrel=1e-10)
            return val
        except Exception:
            return 1e10
    
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_result = x0.copy()
    best_val = objective(x0)
    
    start_points = [
        x0,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]
    
    for sp in start_points:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_result)
