"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Use a hybrid acquisition: combine Expected Improvement with Upper Confidence Bound
    # to balance exploitation and exploration.
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization (lower is better)
    # We want to minimize, so UCB = mu - beta * sigma
    # But we want higher scores to be selected, so we negate it
    # Actually, for minimization, we want to find low values.
    # LCB (Lower Confidence Bound) = mu - beta * sigma gives lower estimates.
    # Points with low LCB are promising for minimization.
    # To make "higher score = evaluate next", we can use -LCB or just use EI.
    
    # Let's try a simple combination: EI + bonus for low sigma (uncertainty reduction)
    # Or use a weighted sum of normalized EI and UCB.
    
    # Normalize EI to [0, 1] roughly for combining
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # UCB for minimization: we prefer low mu and high sigma (exploration)
    # Score = -mu + beta * sigma (higher is better for our "evaluate next" logic)
    # But this might be too exploratory. Let's stick with a modified EI.
    
    # A common alternative is the Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Hybrid: 0.8 * EI + 0.2 * PI (both favor exploitation, PI is more aggressive)
    # Or try: EI with a slight exploration bonus
    # Let's use a simple weighted combination of EI and a normalized UCB score
    
    # For minimization, UCB score (higher is better for selection) = -mu + beta*sigma
    # But we need to scale it. Let's normalize mu and sigma.
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    if mu_max > mu_min:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_norm = np.zeros_like(mu)
        
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # UCB score for minimization: lower mu is better, higher sigma is better for exploration
    # So score_ucb = (1 - mu_norm) + sigma_norm
    score_ucb = (1 - mu_norm) + sigma_norm
    
    # Combine EI and UCB
    # Use 0.7 * EI_norm + 0.3 * score_ucb
    return 0.7 * ei_norm + 0.3 * score_ucb
# EVOLVE-BLOCK-END
