"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a dynamic
    UCB exploration term.
    
    Structural idea from inspiration 1: dynamic beta that shifts from exploration
    to exploitation as f_best improves.
    Refinement machinery from current program: log-transformed EI for compressed
    dynamic range, ensuring the acquisition is less greedy about large EI values
    and gives more relative weight to moderately promising points.
    
    The final score is a blend of log1p(EI) and the dynamic UCB term, where
    the UCB weight decays as f_best decreases (better solutions found).
    
    mu and sigma are in objective units (lower is better for the objective).
    We want to maximize the acquisition score.
    """
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Return a score that favors low mu (exploitation) if we're out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip EI to non-negative for log transform
    ei_clipped = np.maximum(ei, 0.0)

    # Log-transform to compress dynamic range
    score_ei = np.log1p(ei_clipped)

    # Dynamic beta: aggressive exploration early, exploitation late
    # f_best starts high (Ackley can be ~10+) and decreases toward 0
    # When f_best is large (>5), beta ~ 2.5 (explore)
    # When f_best is small (<0.1), beta ~ 0.5 (exploit)
    beta = 0.5 + 2.0 * np.exp(-f_best / 1.5)
    beta = np.clip(beta, 0.3, 3.0)

    # UCB exploration term
    ucb_term = -mu + beta * sigma_safe

    # Blend: primarily log-EI, with a UCB boost for exploration
    # The weight on UCB decays as f_best improves (exploit more late-game)
    ucb_weight = 0.3 * np.exp(-f_best / 3.0)
    ucb_weight = np.clip(ucb_weight, 0.05, 0.5)

    score = score_ei + ucb_weight * ucb_term

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
