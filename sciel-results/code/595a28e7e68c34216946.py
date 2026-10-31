import numpy as np
from scipy.optimize import minimize
import time
from math import gamma

def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    H_nm1 = np.array([1.0])
    H_n = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        H_np1 = 2.0 * np.concatenate([np.zeros(k), H_n]) - 2.0 * (k - 1) * np.concatenate([H_nm1, np.zeros(1)])
        H_nm1, H_n = H_n, H_np1
    return H_n

def construct():
    start_time = time.time()
    deadline = start_time + 190.0
    
    H0 = np.array([1.0])
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)
    
    # Precompute moments
    max_deg = 28  # degree of x^4 * P^2 where P is degree 12
    moments = np.zeros(max_deg + 1)
    for k in range(max_deg + 1):
        if k % 2 == 0:
            moments[k] = gamma((k + 1) / 2.0) / 2.0
        else:
            moments[k] = 0.0
    
    def compute_bound(c0, c4, c8):
        c12 = -(c0 + 12.0 * c4 + 168.0 * c8) / 504.0
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        P2 = np.convolve(P, P)
        I0 = np.dot(P2, moments[:len(P2)])
        I4 = np.dot(np.concatenate([np.zeros(4), P2]), moments[:len(P2)+4])
        if I0 <= 1e-15:
            return 1e10
        return I4 / I0
    
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = compute_bound(*seed)
    best_x = seed.copy()
    
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline:
            break
        x0 = seed.copy() if i == 0 else np.random.randn(3) * 0.3
        try:
            res = minimize(lambda x: compute_bound(x[0], x[1], x[2]), x0,
                          method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x
