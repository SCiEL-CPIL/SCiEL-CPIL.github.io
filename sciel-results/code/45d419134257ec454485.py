import numpy as np

N = 600

def construct():
    # Create indices
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Define a smooth bump function
    # We want to minimize max(f*f) / (max(f) * integral(f))
    # This is equivalent to minimizing integral(f^2) / integral(f) given max(f)=1
    # The triangle function gave 0.66.
    # Let's try a raised cosine (Hann window) on the full interval?
    # f(x) = 0.5 * (1 + cos(2*pi*x / width)) ?
    # Width is 0.5.
    # f(x) = 0.5 * (1 + cos(4*pi*x))
    # At x=0, f=1. At x=0.25, f=0.
    
    f = 0.5 * (1 + np.cos(4 * np.pi * x))
    
    # Ensure non-negative
    f = np.maximum(f, 0)
    
    # Convert to list of floats
    return list(f)
