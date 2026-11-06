import numpy as np

N = 400

def construct():
    best_val = float('inf')
    best_h = None
    
    # Try different frequencies and amplitudes
    # h = 1 + A * sin(2*pi*i/M)
    for M in [N//2, N, 2*N]:
        if M <= 1: continue
        for A in np.linspace(0.5, 2.0, 20):
            h = np.ones(N) + A * np.sin(2 * np.pi * np.arange(N) / M)
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            sum_h = np.sum(h)
            if sum_h == 0:
                continue
            val = abs(2 * N * max_conv / sum_h**2)
            if val < best_val:
                best_val = val
                best_h = h
                
    # Fallback to seed if something went wrong
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
