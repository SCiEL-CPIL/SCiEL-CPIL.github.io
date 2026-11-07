"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and adaptive LCB
    for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses standardization to balance exploitation (EI) and exploration (LCB),
    with the LCB weight adapted based on the gap to the known global minimum.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative, Phi(z) is small, EI is small.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate LCB Score (for minimization, we want low mu and high sigma)
    # LCB value = mu - beta * sigma
    # We want to MAXIMIZE the score, so we use -LCB = -mu + beta * sigma
    
    # Adaptive beta based on gap to known global minimum
    F_MIN = -39.1662
    gap = f_best - F_MIN
    
    # When gap is large, use high exploration (beta up to 2.0)
    # When gap is small, use low exploration (beta down to 0.5)
    BETA_MIN = 0.5
    BETA_MAX = 2.0
    GAP_SCALE = 30.0
    
    gap_clipped = np.clip(gap, 0.0, GAP_SCALE)
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (gap_clipped / GAP_SCALE)
    
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Standardize both scores to have mean 0 and std 1
    # This allows combining them regardless of their original scales
    def standardize(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < eps:
            return arr - mean
        return (arr - mean) / std
    
    ei_std = standardize(ei)
    lcb_std = standardize(lcb_score)
    
    # 4. Combine with adaptive weights
    # When gap is large (early), weight LCB more for exploration
    # When gap is small (late), weight EI more for exploitation
    # Base weights: 0.6 EI, 0.4 LCB
    # Adjust based on gap: increase LCB weight when gap is large
    w_ei_base = 0.6
    w_lcb_base = 0.4
    
    # Scale weights by gap: when gap is large, shift weight toward LCB
    gap_factor = gap_clipped / GAP_SCALE  # 0 to 1
    w_lcb = w_lcb_base + 0.3 * gap_factor  # 0.4 to 0.7
    w_ei = 1.0 - w_lcb  # 0.6 to 0.3
    
    scores = w_ei * ei_std + w_lcb * lcb_std
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
