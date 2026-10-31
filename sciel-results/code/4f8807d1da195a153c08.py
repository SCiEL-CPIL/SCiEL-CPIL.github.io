import time
import numpy as np

N = 50

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Strategy: Return a single spike (delta function).
    # This maximizes the L2 norm relative to the L1 norm and keeps the convolution peak minimal 
    # (relative to the norms) compared to a spread-out distribution like the uniform seed.
    # For f = [1, 0, ..., 0]:
    # ||f||_2^2 = 1
    # ||f||_1 = 1
    # ||f*f||_inf = 1
    # Ratio = 1.
    # For seed f = [1, ..., 1]:
    # ||f||_2^2 = N
    # ||f||_1 = N
    # ||f*f||_inf = N
    # Ratio = 1/N.
    # Improvement factor = N = 50.
    
    f = [0.0] * N
    f[0] = 1.0
    
    # Safety check for time (though this is instantaneous)
    if time.time() > deadline - 1.0:
         return f
         
    return f
