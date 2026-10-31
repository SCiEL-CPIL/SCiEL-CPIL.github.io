import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = 180.0
    
    def hermite_coeffs(n):
        if n == 0: return [1.0]
        if n == 1: return [0.0, 2.0]
        h_prev2 = [1.0]
        h_prev1 = [0.0, 2.0]
        for k in range(2, n+1):
            term1 = [0.0] + h_prev1
            term2 = [-2.0*(k-1)*c for c in h_prev2]
            while len(term2) < len(term1):
                term2.append(0.0)
            h_curr = [t1 + t2 for t1, t2 in zip(term1, term2)]
            h_prev2, h_prev1 = h_prev1, h_curr
        return h_prev1

    H0_c = hermite_coeffs(0)
    H4_c = hermite_coeffs(4)
    H8_c = hermite_coeffs(8)
    H12_c = hermite_coeffs(12)
    
    def pad(c, length):
        return c + [0.0] * (length - len(c))
    
    H0_c = pad(H0_c, 13)
    H4_c = pad(H4_c, 13)
    H8_c = pad(H8_c, 13)
    H12_c = pad(H12_c, 13)
    
    h0_0 = H0_c[0]
    h4_0 = H4_c[0]
    h8_0 = H8_c[0]
    h12_0 = H12_c[0]
    
    sqrt_pi = np.sqrt(np.pi)
    # Precompute moments up to degree 24+2=26
    moments = [0.0] * 27
    for k in range(0, 27, 2):
        m = k // 2
        res = 1.0
        for i in range(1, m+1):
            res *= (2*i - 1) / 2.0
        moments[k] = res * sqrt_pi
    
    def compute_score(c0, c4, c8):
        c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
        
        P_c = c0*H0_c + c4*H4_c + c8*H8_c + c12*H12_c
        
        Q_c = np.convolve(P_c, P_c)
        
        Int0 = 0.0
        Int2 = 0.0
        for k in range(len(Q_c)):
            Int0 += Q_c[k] * moments[k]
            Int2 += Q_c[k] * moments[k+2]
            
        if Int0 == 0: return np.inf
        
        Pd_c = [k*P_c[k] for k in range(1, len(P_c))]
        xP_c = [0.0] + P_c[:-1]
        if len(Pd_c) < len(xP_c):
            Pd_c = Pd_c + [0.0] * (len(xP_c) - len(Pd_c))
        R_c = [a - b for a, b in zip(Pd_c, xP_c)]
        
        S_c = np.convolve(R_c, R_c)
        
        Int_F2 = 0.0
        for k in range(len(S_c)):
            Int_F2 += S_c[k] * moments[k]
            
        val = np.sqrt(Int2 * Int_F2) / Int0
        return val

    seed_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_val = compute_score(seed_c[0], seed_c[1], seed_c[2])
    
    def obj_sphere(params):
        theta, phi = params
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_score(c0, c4, c8)

    if time.time() - start_time > deadline - 10:
        return seed_c

    best_c = seed_c
    best_val = seed_val
    
    try:
        res = differential_evolution(obj_sphere, [(0, np.pi), (0, 2*np.pi)], maxiter=100, tol=1e-8, seed=42, polish=True)
        if res.success:
            theta_opt, phi_opt = res.x
            c0_opt = np.sin(theta_opt) * np.cos(phi_opt)
            c4_opt = np.sin(theta_opt) * np.sin(phi_opt)
            c8_opt = np.cos(theta_opt)
            val_opt = res.fun
            if val_opt < best_val:
                best_val = val_opt
                best_c = np.array([c0_opt, c4_opt, c8_opt])
                
        # Polish
        if time.time() - start_time < deadline - 5:
            res2 = minimize(lambda x: compute_score(x[0], x[1], x[2]), best_c, method='L-BFGS-B', tol=1e-12)
            if res2.success and res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x
                
    except Exception:
        pass

    return best_c
