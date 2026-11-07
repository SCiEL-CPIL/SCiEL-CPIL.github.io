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

    # Dynamic beta: aggressive exploration early, exploitation late
    # f_best starts high (Ackley can be ~10+) and decreases toward 0
    # Map f_best to beta in [0.5, 2.5]
    # When f_best is large (>5), beta ~ 2.5 (explore)
    # When f_best is small (<0.1), beta ~ 0.5 (exploit)
    # Use a smooth transition
    beta = 0.5 + 2.0 * np.exp(-f_best / 1.5)
    beta = np.clip(beta, 0.3, 3.0)

    score = -mu + beta * sigma_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
