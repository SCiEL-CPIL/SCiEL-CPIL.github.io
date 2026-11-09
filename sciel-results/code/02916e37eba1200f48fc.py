"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a combination of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with a temperature parameter that adapts
    based on the uncertainty profile.
    
    This structural change blends exploitation (EI) with exploration (UCB) to
    better balance the search. The UCB component uses a beta parameter that
    increases as uncertainty becomes more concentrated, encouraging exploration
    in high-uncertainty regions early on.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Calculate Expected Improvement
    Z = (f_best - mu) / sigma
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    EI = (f_best - mu) * cdf_Z + sigma * pdf_Z
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # Calculate UCB score: mu - beta * sigma (for minimization, we want low mu, high sigma)
    # Use a beta that adapts to the uncertainty level
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Beta parameter: higher when uncertainty is high relative to mean
    # This encourages exploration when there's significant uncertainty
    beta = 1.0 + 0.5 * (sigma_max / (sigma_mean + 1e-6))
    
    UCB = mu - beta * sigma
    UCB = np.where(np.isfinite(UCB), UCB, 0.0)

    # Blend EI and UCB
    # Normalize both to [0, 1] range for blending
    EI_min, EI_max = np.min(EI), np.max(EI)
    EI_range = EI_max - EI_min + 1e-6
    EI_norm = (EI - EI_min) / EI_range

    UCB_min, UCB_max = np.min(UCB), np.max(UCB)
    UCB_range = UCB_max - UCB_min + 1e-6
    UCB_norm = (UCB - UCB_min) / UCB_range

    # Weight: give more weight to EI when it's informative, UCB when exploring
    # Use a sigmoid-like weight based on the relative magnitude
    weight_ei = 0.7  # Fixed weight for EI, 0.3 for UCB
    scores = weight_ei * EI_norm + (1 - weight_ei) * UCB_norm

    # Scale back to original EI-like magnitude
    scores = scores * (EI_max + 1e-6)
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
