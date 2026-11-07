"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = None
_ACQ_TIME_LIMIT = 290.0 # Safety margin from 300s

def acquisition(mu, sigma, f_best):
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    # Check deadline (unlikely to trigger for this O(N) op, but required)
    if time.time() - _ACQ_START_TIME > _ACQ_TIME_LIMIT:
        # Fallback to random scores if time is up
        return np.random.randn(len(mu))

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Minimization objective: lower is better.
    # UCB for minimization (LCB): mu - beta * sigma
    # We maximize the negative of the LCB, or equivalently:
    # score = -mu + beta * sigma
    
    # Beta controls exploration. 
    # Based on knowledge, lower exploration (smaller beta) was better in a mix.
    # Pure UCB with moderate beta is a good structural alternative to EI.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
