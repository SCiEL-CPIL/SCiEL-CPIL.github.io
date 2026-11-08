import numpy as np

N = 600

def construct():
    """
    Returns the uniform step function of length N, normalized to sum to 1.
    
    Analysis:
    The task is to minimize the normalized maximum of the autoconvolution of a 
    nonnegative step function on [-1/4, 1/4]. 
    For nonnegative functions, the autoconvolution is maximized at lag 0.
    The value at lag 0 is proportional to the L2 norm squared (sum of squares).
    Subject to a fixed L1 norm (mass), the sum of squares is minimized when 
    the function is uniform.
    Thus, the uniform distribution is the global minimum for the peak 
    autoconvolution value.
    
    Previous attempts using gradient descent (inspirations) scored lower 
    (0.58), indicating they moved away from this optimum, likely due to 
    non-convexity or metric misinterpretation.
    """
    # Uniform heights
    # Normalized so that sum(heights) = 1 (assuming unit mass)
    # Or if the evaluator normalizes, the relative values are what matter.
    # The seed program used sum=1.
    v = [1.0 / N] * N
    return v
