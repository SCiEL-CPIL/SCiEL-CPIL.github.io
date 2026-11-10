import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    if p.size != 10:
        if p.size > 10:
            p = p[:10]
        else:
            p = np.pad(p, (0, 10 - p.size))
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Model: sum of 5 Gaussian RBFs
    # y(x) = sum_{i=0}^{4} A_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))
    # Parameters:
    #   p[0:5] -> Amplitudes A_i
    #   p[5:10] -> Means mu_i
    # Widths sigma_i fixed to 1.0 for stability (linear in A_i, smooth in mu_i)
    
    A = p[0:5]
    mu = p[5:10]
    sigma = 1.0
    
    y = np.zeros_like(x)
    for i in range(5):
        diff = x - mu[i]
        y += A[i] * np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Ensure finite output
    y = np.clip(y, -1e150, 1e150)
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
