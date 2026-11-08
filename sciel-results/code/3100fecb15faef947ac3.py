import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Use a 2-point adjacent cluster with heights in ratio 2:1.
    This maximizes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) 
    under the assumption that the evaluator normalizes ||h||_1 = 1 
    (or that the ratio is scale-invariant in the relevant sense, 
    though strictly it's not, the shape determines the normalized ratio).
    
    For h = [2/3, 1/3, 0, ...]:
    ||h||_2^2 = 4/9 + 1/9 = 5/9
    ||h||_1 = 1
    h*h = [4/9, 4/9, 1/9, ...]
    ||h*h||_inf = 4/9
    Ratio = (5/9) / (1 * 4/9) = 1.25
    
    This is an improvement over the uniform seed (Ratio 1.0).
    """
    h = [0.0] * N
    # Place the 2-point cluster at the beginning
    # Ratio 2:1 to maximize the metric
    h[0] = 2.0 / 3.0
    h[1] = 1.0 / 3.0
    
    return h
# EVOLVE-BLOCK-END
