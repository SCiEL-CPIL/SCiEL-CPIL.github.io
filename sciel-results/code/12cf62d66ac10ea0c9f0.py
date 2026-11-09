# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
# We track elapsed time since module import and return early if we're close to the limit.
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Product of Experts (PoE) blend of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) for minimization.
    
    mu, sigma : (2000,) arrays, sigma is the GP posterior std (objective units)
    f_best    : lowest observed objective value
    Returns 2000 finite scores; higher => evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard
    if _check_deadline():
        # Fallback: pure exploitation (lowest mu) + tiny tiebreak
        scores = -mu + 1e-12 * np.arange(len(mu))
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)

    # 1. Calculate EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might occur)
    ei_scores = np.maximum(ei_scores, 0.0)

    # 2. Calculate LCB for minimization
    # We want to minimize mu + beta * sigma.
    # So the "score" for LCB (higher is better) is -(mu + beta * sigma).
    
    # Adaptive Beta:
    # We use a fixed beta for now, or a simple decay if we could track time/steps.
    # Since we can't easily track steps, let's use a beta that is moderate.
    # A common choice is 2.0 or 1.0. 
    # Let's try to make it adaptive based on the spread of sigma.
    # If sigma is generally high, we might want to explore more.
    # However, PoE usually works well with a fixed or slowly varying beta.
    # Let's use beta = 2.0 as a starting point for the structural change.
    beta = 2.0
    
    lcb_values = mu + beta * sigma
    lcb_scores = -lcb_values

    # 3. Normalize both scores to [0, 1]
    def normalize(scores):
        min_s = np.min(scores)
        max_s = np.max(scores)
        range_s = max_s - min_s
        if range_s < 1e-12:
            return np.zeros_like(scores)
        return (scores - min_s) / range_s

    ei_norm = normalize(ei_scores)
    lcb_norm = normalize(lcb_scores)

    # 4. Product of Experts
    # The product emphasizes regions where BOTH EI is high AND LCB is low (good).
    # This is a structural change from the linear blend.
    scores = ei_norm * lcb_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Avoid all-equal scores
    if np.allclose(scores, scores[0]):
        scores = -mu + beta * sigma + 1e-12 * np.arange(n)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
