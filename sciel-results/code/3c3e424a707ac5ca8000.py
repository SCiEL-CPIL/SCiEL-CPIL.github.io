# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: UCB (Upper Confidence Bound) for minimization.
    
    score = -mu + beta * sigma
    
    Uses beta=1.5 to balance exploration and exploitation for the short-horizon
    search on Ackley 2D. This is based on the finding that pure UCB outperforms
    hybrid EI+UCB in this setting.
    """
    # Check deadline (though this function is fast, it's a good practice)
    if time.time() - _t0 > _DEADLINE:
        # Return a random-ish score to break ties if we are running out of time
        # But we must return finite scores.
        # Actually, if we are over deadline, the evaluator kills us.
        # We just return the best we can quickly.
        pass

    sigma_safe = np.maximum(sigma, 1e-8)
    
    beta = 1.5
    score = -mu + beta * sigma_safe
    
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
