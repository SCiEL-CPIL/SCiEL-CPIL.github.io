"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Time guard: kill before the 330s wall-clock limit.
_DEADLINE = time.time() + 325.0


def _time_left():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an ensemble of EI and UCB strategies.
    This combines exploitation (EI) and exploration (UCB) via normalized
    element-wise maximum selection, providing a robust balance without
    arbitrary time-based decay.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _time_left() <= 0:
        # Pure exploitation fallback
        scores = f_best - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    delta = f_best - mu  # positive when mu < f_best (improvement possible)

    # 1. Expected Improvement (EI)
    z = delta / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -10, 10)
    ei_scores = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_scores = np.maximum(ei_scores, 0.0)

    # 2. Upper Confidence Bound (UCB) for minimization
    # Score = beta * sigma - mu (Higher is better)
    # We use two beta values to cover different exploration levels
    
    beta1 = 2.0
    ucb_scores1 = beta1 * sigma_safe - mu
    
    beta2 = 4.0
    ucb_scores2 = beta2 * sigma_safe - mu
    
    # Normalize scores to be comparable across different scales
    # We use Z-score normalization (mean 0, std 1) for each strategy
    def normalize(scores):
        m = np.mean(scores)
        s = np.std(scores)
        if s < 1e-9:
            return scores - m
        return (scores - m) / s
        
    z_ei = normalize(ei_scores)
    z_ucb1 = normalize(ucb_scores1)
    z_ucb2 = normalize(ucb_scores2)
    
    # Combine by taking the element-wise maximum
    # This selects the candidate that is most promising under at least one criterion
    final_scores = np.maximum(z_ei, np.maximum(z_ucb1, z_ucb2))
    
    # Ensure finiteness; non-finite values invalidate the whole result.
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
