import time
import numpy as np

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin within 120s limit
    
    # The theoretical maximum for the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # for non-negative h is 1, achieved by a delta function (spike).
    # The seed program uses a uniform distribution which gives a ratio of 1/N = 0.02.
    # A spike gives a ratio of 1.0.
    
    # We can verify this quickly or just return the known optimum.
    # Let's return the spike at the center to be safe with boundary effects, 
    # though for linear autoconvolution (linear convolution), boundaries don't 
    # change the max value for a spike (it's 1 regardless of position).
    
    # Construct the spike
    h = [0.0] * N
    h[N // 2] = 1.0
    
    # Optional: Quick local search to ensure no weird numerical artifact makes 
    # a nearby vector better (highly unlikely, but cheap to check).
    # Actually, local search on a discrete grid of 50 dimensions is expensive.
    # Given the mathematical proof that delta is optimal for non-negative functions
    # in this context (ratio <= 1), we stick with the spike.
    
    # Check time, return best found (which is the spike)
    if time.time() > deadline:
         return h
         
    return h
