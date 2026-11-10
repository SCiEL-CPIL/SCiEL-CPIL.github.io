# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _calculate_area(x):
    n = 64
    m = 20000
    y = (np.arange(m) + 0.5) / m
    w = (1 - y) / n
    a = (np.arange(n) + 1) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    union_lengths = w[np.newaxis, :] + np.sum(np.minimum(gaps, w[np.newaxis, :]), axis=0)
    return np.mean(union_lengths)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        pos = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        keich_positions.append(pos)
    x_keich = np.array(keich_positions, dtype=np.float64)
    
    best_x = x_keich
    best_area = _calculate_area(x_keich)
    
    rng = np.random.default_rng(42)
    
    # Try Keich with Powell
    res = minimize(_calculate_area, x_keich, method='Powell', 
                   options={'xtol': 1e-8, 'ftol': 1e-10, 'maxiter': 500, 'maxfev': 5000})
    if res.fun < best_area:
        best_area = res.fun
        best_x = res.x
        
    # Random restarts with Powell
    for _ in range(20):
        x_rand = rng.uniform(0, 1, size=n)
        res = minimize(_calculate_area, x_rand, method='Powell', 
                       options={'xtol': 1e-8, 'ftol': 1e-10, 'maxiter': 300, 'maxfev': 3000})
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x
            
    # Final polish with Nelder-Mead on the best found
    res = minimize(_calculate_area, best_x, method='Nelder-Mead', 
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True})
    if res.fun < best_area:
        best_area = res.fun
        best_x = res.x

    return best_x

# EVOLVE-BLOCK-END
