"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata


def acquisition(mu, sigma, f_best):
    """
    Rank-based Hybrid Acquisition Function.
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB) scores.
    Both scores are converted to ranks before combining to ensure scale invariance.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB component: linear in uncertainty
    # Increased beta to 2.5 to encourage more exploration in the multimodal landscape
    beta = 2.5
    ucb = -mu + beta * sigma_safe
    
    # Convert to ranks (higher rank = higher score)
    ei_ranks = rankdata(ei)
    ucb_ranks = rankdata(ucb)
    
    # Normalize ranks to [0, 1] for stable combination
    ei_norm = ei_ranks / len(ei_ranks)
    ucb_norm = ucb_ranks / len(ucb_ranks)
    
    # Weighted combination: 
    # Slightly more weight to UCB to match the increased beta and explore more
    w_ei = 0.5
    w_ucb = 0.5
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
