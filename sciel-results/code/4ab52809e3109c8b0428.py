import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize

N = 50

def construct():
    # We will optimize the first K DCT coefficients
    K = 15
    
    def objective(c):
        # c is a vector of DCT coefficients
        # Compute h
        # idct expects a 1D array. We pad with zeros if K < N
        c_full = np.zeros(N)
        c_full[:K] = c
        h_raw = idct(c_full, type=2, norm='ortho')
        h = np.abs(h_raw)
        
        # Normalize h to fix scale invariance
        # Let's fix sum(h) = 1
        s = np.sum(h)
        if s == 0:
            return 0
        h = h / s
        
        # Calculate g
        g = np.convolve(h, h)
        m = len(g)
        
        # Construct y
        y = np.zeros(m + 2)
        y[1:-1] = g
        
        # Calculate l2
        w = 1.0 / (m + 1)
        # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
        # Vectorized calculation
        y_sq = y**2
        y_prod = y[:-1] * y[1:]
        l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_prod + y_sq[1:])
        
        # Calculate norms
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
            return 0
            
        value = l2 / (norm_1 * norm_inf)
        
        # We want to maximize value, so minimize negative value
        return -value

    # Initial guess: uniform h corresponds to only c[0] being non-zero
    # For ortho DCT, c[0] = sqrt(2/N) * sum(h) * ... 
    # Let's just start with a small random vector or specific values
    # Uniform h=1/N -> sum=1.
    # DCT of uniform vector: c[0] is large, others 0.
    # c[0] = sqrt(2/N) * (1/N) * N ? No.
    # idct(c) = h.
    # If h is uniform 1/N, then c[0] = sqrt(2/N) * sum(h) * 1/sqrt(2)? 
    # Let's just use a random start or a known good start.
    
    # Let's try starting with a Gaussian-like DCT spectrum
    # Or just random
    np.random.seed(42)
    c0 = np.random.randn(K) * 0.1
    # Force c[0] to be positive and dominant to ensure non-negativity mostly
    c0[0] = 1.0
    
    res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    c_opt = res.x
    c_full = np.zeros(N)
    c_full[:K] = c_opt
    h_raw = idct(c_full, type=2, norm='ortho')
    h = np.abs(h_raw)
    
    # Final normalization
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = [1.0/N] * N
        
    return h
