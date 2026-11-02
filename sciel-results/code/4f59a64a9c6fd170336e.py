import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def _hermite_phys(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_nm1 = np.ones_like(x)
    H_n = 2 * x
    for k in range(2, n + 1):
        H_np1 = 2 * x * H_n - 2 * (k - 1) * H_nm1
        H_nm1 = H_n
        H_n = H_np1
    return H_n

def _precompute_data():
    max_n = 13
    coeffs = [[0.0] * (n + 1) for n in range(max_n + 1)]
    coeffs[0][0] = 1.0
    coeffs[1][1] = 2.0
    for n in range(2, max_n + 1):
        for k in range(n + 1):
            val = 0.0
            if k - 1 >= 0:
                val += 2.0 * coeffs[n-1][k-1]
            if n - 2 >= 0 and k <= n - 2:
                val -= 2.0 * (n - 1) * coeffs[n-2][k]
            coeffs[n][k] = val
            
    M_n = [0.0] * 27
    for n in range(27):
        if n % 2 == 1:
            M_n[n] = 0.0
        else:
            k = n // 2
            M_n[n] = factorial(2*k) / factorial(k) * np.sqrt(np.pi / 2.0)
            
    return coeffs, M_n

def _compute_integrals_fast(c0, c4, c8, c12, coeffs, M_n):
    p_coeffs = [0.0] * 13
    for k in range(13):
        p_coeffs[k] = c0 * coeffs[0][k] + c4 * coeffs[4][k] + c8 * coeffs[8][k] + c12 * coeffs[12][k]
        
    q_coeffs = [0.0] * 14
    for k in range(13):
        if k + 1 < 13:
            q_coeffs[k] += (k + 1) * p_coeffs[k + 1]
        if k < 14:
            q_coeffs[k + 1] -= 0.5 * p_coeffs[k]
            
    def calc_int(p1, p2, m):
        total = 0.0
        max_deg1 = len(p1) - 1
        max_deg2 = len(p2) - 1
        for i in range(max_deg1 + 1):
            if p1[i] == 0: continue
            for j in range(max_deg2 + 1):
                if p2[j] == 0: continue
                n = i + j + m
                if n < 27:
                    total += p1[i] * p2[j] * M_n[n]
        return total
        
    N = calc_int(p_coeffs, p_coeffs, 0)
    M2 = calc_int(p_coeffs, p_coeffs, 2)
    K = calc_int(q_coeffs, q_coeffs, 0)
    
    if N <= 0:
        return float('inf')
        
    return (M2 * K) / (N * N)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin for 180s limit
    
    coeffs, M_n = _precompute_data()
    
    # Seed
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
        return _compute_integrals_fast(c0, c4, c8, c12, coeffs, M_n)
        
    best_c = np.array([c0_seed, c4_seed, c8_seed])
    best_val = objective(best_c)
    
    # Multi-start strategy
    # 1. Seed
    starts = [np.array([c0_seed, c4_seed, c8_seed])]
    
    # 2. Random perturbations around seed
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.1, size=3)
        starts.append(np.array([c0_seed, c4_seed, c8_seed]) + noise)
        
    # 3. Some other reasonable points? Maybe scaled versions?
    starts.append(np.array([c0_seed * 0.5, c4_seed * 0.5, c8_seed * 0.5]))
    starts.append(np.array([c0_seed * 1.5, c4_seed * 1.5, c8_seed * 1.5]))
    
    for x0 in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B for speed and precision
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
            
            if time.time() > deadline - 5:
                continue
                
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
                
            # Polish with tighter tolerances if we improved
            if time.time() < deadline - 10:
                res2 = minimize(objective, best_c, method='BFGS', 
                                options={'gtol': 1e-15, 'disp': False, 'maxiter': 100})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_c = res2.x
                    
        except Exception:
            continue
            
    return np.array(best_c)
