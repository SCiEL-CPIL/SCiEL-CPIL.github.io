import numpy as np

# Global counter for adaptive beta
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Increment counter
    _eval_count += 1
    
    # Adaptive beta: sqrt(2 * log(t))
    # t starts at 1 for the first call after init?
    # The evaluator starts with 5 points.
    # So the first call to acquisition happens after 5 points are evaluated.
    # So t should be 5 + (number of calls so far).
    # Let's initialize _eval_count to 5?
    # The prompt says: "evaluator starts from 5 uniform random points, then at each step... calls acquisition"
    # So the first call is for the 6th evaluation.
    # We can assume _eval_count starts at 5.
    
    if _eval_count == 0:
        _eval_count = 5 # Initialize to 5, as 5 points are already sampled
        
    t = _eval_count
    beta = np.sqrt(2 * np.log(t + 1))
    
    # UCB for minimization: maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    return score
