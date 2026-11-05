import numpy as np
from scipy.optimize import minimize

N = 400

def objective(params):
    h = params
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    
    c = np.convolve(h, h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)

def construct():
    best_val = 2.0
    best_h = np.ones(N)
    
    # Perform random restarts
    # Include the known optimal solution (uniform) as a start
    starts = [np.ones(N)]
    
    for _ in range(10):
        # Random Gaussian
        starts.append(np.random.randn(N))
        
        # Smooth noise
        freqs = np.fft.rfftfreq(N)
        spectrum = np.exp(-freqs * 5) * np.random.randn(len(freqs))
        starts.append(np.fft.irfft(spectrum, N))
        
        # Sine wave
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.sin(t))
        
    for h_start in starts:
        if abs(np.sum(h_start)) < 1e-6:
            h_start += 0.1
            
        res = minimize(objective, h_start, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_h = res.x
            
    return best_h
