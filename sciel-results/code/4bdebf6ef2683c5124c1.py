import numpy as np

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def construct():
    # Create two sharp, disjoint Gaussian pulses
    # Centers aligned with grid points for maximum sharpness
    # Grid points: -0.245, -0.235, ..., -0.205 (idx 4), ..., 0.205 (idx 45), ...
    
    c1 = -0.205
    c2 = 0.205
    s = 0.001 # Very small width
    
    w1 = np.exp(-0.5 * ((x - c1) / s) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s) ** 2)
    
    a = 1.0
    b = 1.0
    
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
