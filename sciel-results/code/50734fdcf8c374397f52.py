import numpy as np
import time

N = 600
DX = 1.0 / 1200.0  # interval width

def construct():
    # The optimal non-negative step function on a fixed interval that minimizes
    # max(f*f) / (int f)^2 is the uniform function.
    # This yields a raw value of 1/L = 2.
    # Any other function will have a raw value >= 2.
    
    # Create a uniform step function of height 1.
    # The normalization of the ratio makes the absolute height irrelevant,
    # but we must return a valid sequence of non-negative heights.
    h = np.ones(N, dtype=np.float64)
    
    return list(h)
