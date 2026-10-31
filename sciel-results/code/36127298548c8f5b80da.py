import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    The seed program used a uniform box function, which corresponds to
    maximizing the overlap at lag 0. However, by using a sparse function
    (e.g., alternating 1s and 0s), we minimize the peak of the autoconvolution
    relative to the norm of the function.
    
    For a function consisting of K disjoint boxes of width w and height h,
    the peak of the autoconvolution is approximately h^2 * w.
    The L2 norm squared is K * h^2 * w.
    The ratio is 1/K.
    To minimize the ratio, we maximize K.
    With N intervals, the maximum number of disjoint boxes is N/2 
    (by setting every other interval to 0).
    This yields a ratio of ~2/N, which is significantly lower than 
    the uniform box ratio of 1 (for L2 norm) or 2 (for L1 norm squared).
    """
    start_time = time.time()
    deadline = 145.0
    
    # Create a sparse step function: 1 at even indices, 0 at odd indices.
    # This creates N/2 disjoint "boxes".
    heights = np.zeros(N)
    heights[::2] = 1.0
    
    # Check time
    if time.time() - start_time > deadline:
        return heights.tolist()
        
    return heights.tolist()
# EVOLVE-BLOCK-END
