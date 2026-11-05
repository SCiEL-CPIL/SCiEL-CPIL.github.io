import numpy as np

N = 400

def construct():
    # Try a real chirp sequence
    # h_i = cos(pi * i^2 / N)
    i = np.arange(N)
    h_chirp = np.cos(np.pi * i**2 / N)
    
    s = np.sum(h_chirp)
    if abs(s) < 1e-6:
        # If sum is zero, add a small constant or use a different phase
        # Try h_i = cos(pi * i^2 / N + pi/4)
        h_chirp = np.cos(np.pi * i**2 / N + np.pi/4)
        s = np.sum(h_chirp)
        
    if abs(s) < 1e-6:
        # Fallback to seed if sum is still zero
        return np.ones(N)
        
    h_opt = h_chirp / s
    
    val = abs(2 * N * np.max(np.convolve(h_opt, h_opt)) / np.sum(h_opt)**2)
    
    # Compare with seed
    seed_val = 2.0
    
    if val < seed_val:
        return h_opt
    else:
        return np.ones(N)
