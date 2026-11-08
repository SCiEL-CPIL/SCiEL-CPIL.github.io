import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # We search for a probability vector p (sum=1, p>=0)
    # that minimizes max(conv(p, p))
    
    # Parameterization:
    # p_i = (1/N) * (1 + sum_{k=1}^K A_k cos(2*pi*k*i/N) + B_k sin(2*pi*k*i/N))
    # We use K=2 harmonics. 4 parameters.
    
    def get_p(params):
        # params: [A1, B1, A2, B2]
        i = np.arange(N)
        p = np.ones(N)
        for k in range(1, 3):
            p += params[2*(k-1)] * np.cos(2 * np.pi * k * i / N)
            p += params[2*(k-1)+1] * np.sin(2 * np.pi * k * i / N)
        p /= N
        return p

    def objective(params):
        p = get_p(params)
        # Check non-negativity
        if np.any(p < 0):
            return 1e9
        # Normalize just in case (should be sum 1)
        s = np.sum(p)
        if s == 0:
            return 1e9
        p = p / s
        
        # Compute convolution
        c = np.convolve(p, p, mode='full')
        return c.max()

    # Initial guesses
    best_p = np.ones(N) / N
    best_val = np.convolve(best_p, best_p, mode='full').max()
    
    # Try a few random starts
    starts = [
        np.zeros(4),
        np.array([0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0, 0.0]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.2, -0.1, 0.0, 0.0]),
        np.array([-0.2, 0.1, 0.0, 0.0]),
        np.random.uniform(-0.2, 0.2, 4)
    ]
    
    for x0 in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'maxiter': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_p = get_p(res.x)
                # Ensure non-negative and normalized
                best_p = np.clip(best_p, 0, None)
                best_p = best_p / np.sum(best_p)
        except Exception:
            continue

    # If we found a better one, return it
    # The return value is the sequence of heights h
    # The seed returned h such that sum(h) was used for normalization.
    # If we return p (sum=1), the normalization is by 1.
    # This is consistent.
    
    return [float(x) for x in best_p]
