"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = time.time() + 320  # 320s wall-clock safety margin (evaluator kills at 330s)
_CALL_START = [time.time()]


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _CALL_START[0]) > 0.5


def acquisition(mu, sigma, f_best):
    """
    Acquisition function blending Expected Improvement and Upper Confidence Bound.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior standard deviation in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        array of shape (2000,), scores where higher means evaluate next
    """
    _CALL_START[0] = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB) - for minimization, we want low mu + beta*sigma
    # Higher UCB score means more promising
    # beta controls exploration; adapt based on how far we are from optimum
    # Weierstrass min is 0, so if f_best is close to 0, explore less
    # Use a decay: beta starts high and decreases as f_best approaches 0
    beta = 0.5 * (1.0 + np.exp(-f_best * 5.0))  # beta in [0.5, 1.0]
    ucb_score = -(mu + beta * sigma)  # negative because lower is better for objective
    
    # Normalize both to have comparable scales
    # EI is non-negative, UCB_score can be any real
    # Blend: weight EI more when we have good estimates, UCB when uncertain
    # Use a fixed blend that was measured to work well
    weight_ei = 0.7
    weight_ucb = 0.3
    
    # Normalize UCB to be on similar scale as EI
    # EI ranges from 0 to ~|f_best|, UCB ranges broadly
    # Scale UCB by typical EI magnitude
    ei_scale = np.mean(ei) + 1e-9
    ucb_normalized = ucb_score / (ei_scale + 1e-9)
    
    # Final score: higher is better
    score = weight_ei * ei + weight_ucb * np.maximum(ucb_normalized, 0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
