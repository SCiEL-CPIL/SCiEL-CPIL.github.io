import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre
from scipy.integrate import quad
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590.0
    
    try:
        roots = roots_genlaguerre(12, 1)[0]
    except Exception:
        roots = np.array([4.0 * (i + 1) for i in range(12)])
        
    roots = np.sort(np.abs(roots))
    
    best_score = np.inf
    best_roots = roots
    
    def objective(root_vec):
        if time.time() > deadline:
            return 1e100
        
        x_roots = root_vec
        if np.any(x_roots <= 0):
            return 1e100
        
        coeffs = np.poly(x_roots)
        
        # x-space moments
        nodes, weights = np.polynomial.laguerre.laggauss(50)
        P_vals = np.polyval(coeffs, nodes)
        f2 = np.sum(weights * P_vals**2)
        xf2 = np.sum(weights * nodes**2 * P_vals**2)
        
        if f2 <= 0:
            return 1e100
        sigma_x2 = xf2 / f2
        
        # k-space moments
        # F(k) = sum_{m=0}^{12} coeffs[12-m] * m! / (0.5 + 2j*pi*k)^(m+1)
        facts = np.array([math.factorial(m) for m in range(13)])
        
        def F_k(k):
            val = 0.0
            for m in range(13):
                c_m = coeffs[12-m]
                if c_m != 0:
                    val += c_m * facts[m] / (0.5 + 2j * np.pi * k) ** (m + 1)
            return val
        
        # Integrate |F(k)|^2
        # Use quad for accuracy
        def integrand_F2(k):
            fk = F_k(k)
            return np.abs(fk)**2
            
        def integrand_kF2(k):
            fk = F_k(k)
            return k**2 * np.abs(fk)**2
            
        try:
            F2, _ = quad(integrand_F2, 0, np.inf, limit=100)
            kF2, _ = quad(integrand_kF2, 0, np.inf, limit=100)
        except Exception:
            return 1e100
            
        if F2 <= 0:
            return 1e100
        sigma_k2 = kF2 / F2
        
        return np.sqrt(sigma_x2 * sigma_k2)

    # Optimize
    try:
        res = minimize(objective, best_roots, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() < deadline and res.fun < best_score:
            best_score = res.fun
            best_roots = res.x
            best_roots = np.sort(np.abs(best_roots))
    except Exception:
        pass
        
    return [float(r) for r in best_roots]
