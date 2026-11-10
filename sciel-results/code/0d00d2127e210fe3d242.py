import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    def area(x):
        # x shape (n,)
        # L shape (n, M)
        L = x[:, np.newaxis] + a[np.newaxis, :] * y[np.newaxis, :]
        w = (1 - y) / n
        
        L_sorted = np.sort(L, axis=0)
        gaps = L_sorted[1:, :] - L_sorted[:-1, :]
        
        # w is shape (M,), need to broadcast to (1, M)
        # gaps is (n-1, M)
        min_gaps = np.minimum(gaps, w[np.newaxis, :])
        
        union_lengths = w + np.sum(min_gaps, axis=0)
        
        return np.mean(union_lengths)

    x0 = np.zeros(n)
    
    # L-BFGS-B is robust for this
    res = minimize(area, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
    
    return res.x
