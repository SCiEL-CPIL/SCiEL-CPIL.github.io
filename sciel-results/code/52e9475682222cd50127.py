# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Deadline guard: 600s limit, keep 30s margin
    deadline = start_time + 570
    
    # We identified that S={2} with f(2)=-2, f(1)=1 is feasible and yields a score of ~0.693.
    # This is better than the previous failure (0.0).
    # To potentially beat the seed (0.92), we might need a more complex set, but 
    # ensuring feasibility and non-zero score is the first priority.
    
    # Domain: [1, 2]
    domain = np.array([1, 2], dtype=np.int64)
    
    # Values:
    # f(1) = 1.0
    # f(2) = -2.0
    # Check: sum f(k)/k = 1/1 + (-2)/2 = 0. Correct.
    # M(x) = 1*x - 2*floor(x/2).
    # x=1: 1 - 0 = 1 <= 1.0001.
    # x=2: 2 - 2 = 0.
    # x=3: 3 - 2 = 1.
    # Max M(x) = 1. Feasible.
    # a = - (1*log(1)/1 + (-2)*log(2)/2) = - (0 - log(2)) = log(2) ~ 0.693.
    
    values = np.array([1.0, -2.0], dtype=np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
