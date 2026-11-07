# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Return a score that favors low mu (exploitation) if we're out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)

    # Thompson Sampling: draw a single posterior sample for each candidate
    # theta = mu + sigma * Z, where Z ~ N(0, 1)
    # Score = -theta (lower theta is better, so higher score = lower theta)
    rng = np.random.default_rng()
    Z = rng.standard_normal(len(mu))
    theta = mu + sigma_safe * Z
    
    # Score: we want to maximize, so negate theta (since we minimize Ackley)
    score = -theta

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
