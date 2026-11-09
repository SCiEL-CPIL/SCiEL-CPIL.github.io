import numpy as np
from scipy.optimize import minimize

N = 100

def compute_max_overlap(h):
    # h is array of length N
    max_ov = 0.0
    for k in range(1, N):
        # Shift h by k
        # Overlap sum: sum_i h[i] * (1 - h[i+k]) for i+k < N
        # Plus sum_i h[i] for i+k >= N (since h[i+k] is 0)
        
        # Vectorized calculation
        # Part 1: i from 0 to N-1-k
        # h[i] * (1 - h[i+k])
        if N - k > 0:
            part1 = np.sum(h[:N-k] * (1 - h[k:]))
        else:
            part1 = 0.0
            
        # Part 2: i from N-k to N-1
        # h[i] * (1 - 0) = h[i]
        if k > 0:
            part2 = np.sum(h[N-k:])
        else:
            part2 = 0.0
            
        ov = part1 + part2
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    # Initial guess: Sine squared taper
    # h_i = sin^2(pi * i / N)
    # Sum is approx N/2
    i = np.arange(N)
    h_init = np.sin(np.pi * i / N)**2
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(h_init)
    if current_sum > 0:
        h_init = h_init * (N / 2.0) / current_sum
    
    # Clip to [0, 1]
    h_init = np.clip(h_init, 0.0, 1.0)
    
    # Re-normalize after clipping? 
    # If clipping changed the sum, we need to adjust.
    # Simple iterative projection:
    for _ in range(10):
        h_init = np.clip(h_init, 0.0, 1.0)
        s = np.sum(h_init)
        if s == 0: break
        h_init = h_init * (N / 2.0) / s

    # Local optimization using Nelder-Mead on the vector h?
    # Too slow for 100 dims.
    # Try parametric optimization:
    # h_i = 0.5 + A * cos(2*pi*i/N) + B * cos(4*pi*i/N)
    
    def obj(params):
        A, B = params
        h = 0.5 + A * np.cos(2 * np.pi * i / N) + B * np.cos(4 * np.pi * i / N)
        h = np.clip(h, 0.0, 1.0)
        # Normalize sum
        s = np.sum(h)
        if s == 0: return 1e9
        h = h * (N / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
        # Re-normalize again to be safe
        s2 = np.sum(h)
        if s2 == 0: return 1e9
        h = h * (N / 2.0) / s2
        return compute_max_overlap(h)

    # Start from sine squared: A=-0.5, B=0
    res = minimize(obj, x0=[-0.5, 0.0], method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    
    h_opt = 0.5 + res.x[0] * np.cos(2 * np.pi * i / N) + res.x[1] * np.cos(4 * np.pi * i / N)
    h_opt = np.clip(h_opt, 0.0, 1.0)
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt * (N / 2.0) / s
    
    return h_opt
