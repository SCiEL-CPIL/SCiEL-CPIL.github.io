import time
import sys
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize
from numpy.polynomial import polynomial as P

def construct():
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit
    
    # Alpha for generalized Laguerre polynomials
    alpha = -0.5
    max_k = 25
    
    # Precompute Laguerre polynomials L_2k^(alpha)(x) and their derivatives
    # Store coefficients (highest to lowest)
    L_coeffs = []
    Lp_coeffs = []
    for k in range(max_k + 1):
        n = 2 * k
        L = genlaguerre(n, alpha)
        L_coeffs.append(L.coeffs)
        Lp = L.derivative()
        Lp_coeffs.append(Lp.coeffs)
        
    def estimate_B(zs):
        # zs: list of 12 floats
        if len(zs) != 12:
            return 1e10
        if any(z <= 0 for z in zs):
            return 1e10
        if len(set(np.round(zs, 10))) != 12:
            return 1e10
            
        # Build A and b
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            # Evaluate L_2k(0)
            # L_coeffs[k] is highest to lowest.
            # Constant term is L_coeffs[k][-1]
            A[0, k] = L_coeffs[k][-1]
        b[0] = 0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            # Evaluate L'_2k(0)
            A[1, k] = Lp_coeffs[k][-1]
        b[1] = 1
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        for i, z in enumerate(zs):
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            for k in range(26):
                # Evaluate polynomial at z
                # np.polyval takes highest to lowest
                A[row_g, k] = np.polyval(L_coeffs[k], z)
                A[row_gp, k] = np.polyval(Lp_coeffs[k], z)
            b[row_g] = 0
            b[row_gp] = 0
            
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Construct g(x) coefficients (highest to lowest)
        g_coeffs = np.zeros(51)
        for k in range(26):
            deg = 2 * k
            start_idx = 50 - deg
            g_coeffs[start_idx:] += c[k] * L_coeffs[k]
            
        # Construct D(x) = x * prod (x - z_i)^2
        D_coeffs = np.array([1.0, 0.0])
        for z in zs:
            factor = np.array([1.0, -2*z, z*z])
            D_coeffs = np.convolve(D_coeffs, factor)
            
        try:
            q_coeffs, remainder = np.polydiv(g_coeffs, D_coeffs)
        except:
            return 1e10
            
        try:
            roots = np.roots(q_coeffs)
        except:
            return 1e10
            
        real_roots = []
        for r in roots:
            if np.isreal(r):
                rv = np.real(r)
                if rv > 1e-6:
                    real_roots.append(rv)
                    
        if not real_roots:
            return 1e10
            
        return max(real_roots)
        
    # Initial guess
    z0 = np.array([2.0 * (i + 1) for i in range(12)])
    
    best_z = z0.copy()
    best_score = estimate_B(best_z)
    
    # Optimization
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        if r == 0:
            z_current = z0.copy()
        else:
            z_current = np.sort(np.random.uniform(0.5, 20.0, 12))
            while len(set(np.round(z_current, 10))) < 12:
                z_current = np.sort(np.random.uniform(0.5, 20.0, 12))
                
        def objective(z):
            return estimate_B(z)
            
        try:
            res = minimize(objective, z_current, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 100})
            if res.fun < best_score:
                best_score = res.fun
                best_z = res.x
        except Exception:
            pass
            
    result = sorted([float(z) for z in best_z])
    
    if len(set(np.round(result, 10))) != 12 or any(z <= 0 for z in result):
        return [2.0 * (i + 1) for i in range(12)]
        
    return result
