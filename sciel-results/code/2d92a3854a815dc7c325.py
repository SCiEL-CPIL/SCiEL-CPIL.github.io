import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Use a constant step function.
    # For the metric (max autoconvolution) / (sum of heights)^2,
    # a constant function on the entire domain yields a score of 1/N.
    # A delta function yields a score of 1.
    # 1/600 is significantly lower than 1, and likely lower than the
    # current score if the current score is indeed ~1/N or similar.
    # Even if the current score is anomalous, the constant function
    # is a fundamental extremal candidate for this type of problem.
    
    f = np.ones(N)
    
    return f.tolist()
# EVOLVE-BLOCK-END
