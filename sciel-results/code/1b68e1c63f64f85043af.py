"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We track elapsed time to ensure we finish within the deadline.
# The evaluator kills after 330s. We want to be safe.
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Strategy: 
    1. Compute Expected Improvement (EI).
    2. Compute a "Distance-to-Best" penalty/weight. 
       Since we don't have the best point's coordinates directly passed, 
       we cannot compute distance to the best point in the input space directly 
       from mu/sigma alone. 
       
       However, the prompt says: "No coordinates are passed." 
       Wait, the prompt says: "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
       
       This is a crucial constraint. I cannot compute distance to the best point 
       because I don't know the coordinates of the 2000 candidates, nor the 
       coordinates of the best point. 
       
       Therefore, the "Distance-to-Best" strategy suggested in the directive 
       is NOT directly implementable with just mu and sigma unless I make 
       assumptions or use a different mechanism.
       
       Let's re-read the directive: 
       "Strategy to try in this change: Use a 'Distance-to-Best' weighted acquisition 
       that scales the Expected Improvement by the inverse of the distance to the 
       current best point in the standardized input space..."
       
       If I cannot access coordinates, I cannot compute this distance. 
       
       However, I can use a different structural change. 
       The knowledge suggests Thompson Sampling with adaptive exploitation bias 
       improved scores. 
       
       Let's try a hybrid approach: 
       1. Expected Improvement (EI) for exploitation.
       2. Add a stochastic exploration term based on the uncertainty (sigma) 
          to encourage visiting high-variance regions, which often correspond 
          to unexplored areas. 
       
       Actually, a common effective strategy for multimodal functions like 
       Michalewicz is to use a combination of EI and a penalty for being too 
       close to already sampled points. But again, no coordinates.
       
       Alternative: Use a "Lower Confidence Bound" (LCB) style acquisition, 
       which naturally balances exploration and exploitation. 
       LCB = mu - kappa * sigma
       
       For minimization, we want to minimize mu and maximize sigma (uncertainty). 
       So LCB = mu - kappa * sigma is a good candidate. 
       
       Let's try a dynamic kappa that starts high (exploration) and decreases 
       (exploitation) over time. But we don't have the iteration number. 
       
       We can estimate the "stage" of the optimization by looking at f_best. 
       If f_best is close to the minimum (-1.8013), we should exploit more. 
       If f_best is far, we should explore more. 
       
       Let's define a target minimum for Michalewicz 2D as -1.8013. 
       Let progress = (f_best - (-1.8013)) / (initial_guess - (-1.8013)). 
       But we don't know the initial guess. 
       
       Simpler: Use a fixed kappa that works well for Michalewicz. 
       Or, use a combination of EI and a pure exploration term. 
       
       Let's try: 
       score = EI + beta * sigma
       
       Where beta controls the exploration-exploitation trade-off. 
       A high beta encourages exploring high-uncertainty regions. 
       
       To make it adaptive, we can scale beta based on how close f_best is to 
       the known minimum. 
       
       Let min_val = -1.8013
       Let gap = f_best - min_val
       Let beta = max(0.1, 1.0 / (1.0 + gap))
       
       This way, when f_best is far from min_val (large gap), beta is small, 
       favoring EI (exploitation of known good areas). 
       When f_best is close to min_val (small gap), beta is larger, 
       favoring exploration to find even better points. 
       
       Wait, that's backwards. 
       When we are far from the optimum, we should explore more to find the 
       basin of attraction. 
       When we are close, we should exploit to fine-tune. 
       
       So: 
       If gap is large -> high beta (explore)
       If gap is small -> low beta (exploit)
       
       beta = max(0.1, min(2.0, gap))
       
       Let's try this. 
       
       Also, we must ensure the function is fast. 
       """
    
    # Check deadline
    if time.time() - _start_time > _DEALINE:
        # If we're over deadline, just return EI to be safe and fast
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration term
    min_val = -1.8013
    gap = f_best - min_val
    # Scale beta: larger gap -> more exploration
    # gap is typically positive (f_best > min_val)
    # If f_best is -1.7, gap = 0.1
    # If f_best is -1.0, gap = 0.8
    beta = np.clip(gap, 0.1, 2.0)
    
    # Combine EI and exploration
    # We want to maximize the score
    score = ei + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
