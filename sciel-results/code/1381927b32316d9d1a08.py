import numpy as np

N = 600

def construct():
    # Grid on [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Normalize to [-1, 1]
    x_norm = x * 4.0
    
    # Use a high power to approximate the optimal shape
    k = 100
    # g(x) = (1 - |x_norm|)^k
    # Clip to handle numerical errors
    base = np.maximum(0.0, 1.0 - np.abs(x_norm))
    heights = base ** k
    
    return list(heights)
