<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf
=======
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def local_search_polish(h, max_iter=1000, tol=1e-10):
    """Local search: try moving mass between pairs of indices to improve ratio."""
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h
    h = h / l1
    
    best_score = compute_ratio(h)
    support = np.where(h > 1e-10)[0]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving mass from one index to another
        for i in support:
            for j in range(N):
                if i == j or h[j] < 1e-10 and h[i] < 1e-10:
                    continue
                
                # Try transferring small amount from i to j
                for frac in [0.1, 0.05, 0.01]:
                    h_trial = h.copy()
                    amount = h[i] * frac
                    h_trial[i] -= amount
                    h_trial[j] += amount
                    
                    if h_trial[i] < 0 or h_trial[j] < 0:
                        continue
                    
                    score = compute_ratio(h_trial)
                    if score > best_score + tol:
                        h = h_trial
                        best_score = score
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return h
>>>>>>> REPLACE

