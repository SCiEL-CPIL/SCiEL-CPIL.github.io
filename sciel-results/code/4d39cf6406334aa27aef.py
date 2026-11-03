import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 115  # safety margin below 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of 50 nonnegative heights.
    Strategy: Return a vector close to a delta function (one large value, rest small/zero)
    to approach the theoretical upper bound of the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    The inequality ||h||_2^2 <= ||h||_1 * ||h*h||_inf suggests the ratio is bounded by 1.
    A delta function h = [1, 0, ..., 0] gives:
    ||h||_2^2 = 1
    ||h||_1 = 1
    h*h = [1, 0, ..., 0] (assuming circular or linear conv? Task says autoconvolution on [-1/4, 1/4] intervals, likely discrete conv)
    If linear conv of [1,0,..] is [1,0,..], max is 1. Ratio = 1/(1*1) = 1.
    
    The seed [1,1,...,1] gives:
    ||h||_2^2 = N
    ||h||_1 = N
    Conv is a triangle. Max of conv of N ones is N.
    Ratio = N / (N * N) = 1/N.
    For N=50, ratio is 0.02.
    
    So a delta function is a huge improvement.
    To be safe against numerical issues or specific definitions (e.g., if "nonnegative" implies >0),
    we use a very steep geometric decay or just [1.0] + [0.0]*49.
    However, sometimes "heights" in such contexts might imply a probability distribution or positive density.
    But "nonnegative" explicitly allows 0.
    
    Let's verify the "supremum norm of autoconvolution".
    If the convolution is defined on the same grid of 50 points (circular convolution),
    h = [1,0,...,0]. Conv is [1,0,...,0]. Max 1.
    
    If linear convolution, length is 99. The evaluator likely projects back or just takes the max over the defined interval.
    Usually, for "heights on intervals", the convolution is also on intervals.
    
    Let's try a vector that is [1.0] followed by zeros.
    """
    # Ensure we are within time limit
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] + [0.0] * (N - 1)

    # The optimal theoretical bound is 1.0.
    # A single impulse achieves exactly 1.0 (assuming standard definitions).
    h = np.zeros(N)
    h[0] = 1.0
    
    # To be robust against potential strict positivity requirements (unlikely given "nonnegative"),
    # or to explore if there's any weirdness, we could add a tiny epsilon to others,
    # but that would lower the ratio.
    # Ratio for h=[1, eps, ...]:
    # ||h||_2^2 = 1 + 49 eps^2
    # ||h||_1 = 1 + 49 eps
    # Conv max: h[0]*h[0] + h[1]*h[1]? No, h*h[0] = sum h[i]h[i+k].
    # For k=0: sum h[i]^2 = 1 + 49 eps^2.
    # Max conv is likely at k=0.
    # Ratio = (1 + 49 eps^2) / ((1 + 49 eps) * (1 + 49 eps^2)) = 1 / (1 + 49 eps).
    # This is < 1.
    # So pure zero is better.
    
    return list(h)
# EVOLVE-BLOCK-END
