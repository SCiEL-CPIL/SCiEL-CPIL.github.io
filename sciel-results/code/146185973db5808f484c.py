"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of Expected Improvement
    and Upper Confidence Bound, normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
        # Fallback: return something simple and fast
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for EI
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = improvement * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # UCB term: mu - beta * sigma (for minimization, we want low mu and high sigma)
    # Actually for minimization, UCB would be mu + beta * sigma but we want to minimize
    # So we use -mu + beta * sigma as the "upper confidence bound" for improvement potential
    # Or more standard: we want to minimize, so UCB for minimization is mu + beta*sigma
    # But acquisition should be higher for more promising, so we want - (mu + beta*sigma) + bonus
    
    # Let's use a hybrid: combine EI with a UCB-like exploration term
    # UCB for minimization: the "worst case" value is mu + beta * sigma
    # We want to minimize this, so the score contribution is - (mu + beta * sigma)
    # But this doesn't directly help with improvement. Let's think differently.
    
    # Alternative: Use PI (Probability of Improvement) combined with EI
    pi = Phi  # P(improvement) = P(f(x) < f_best) = Phi(z)
    
    # Combined acquisition: weight EI and PI and uncertainty
    # Score = (w1 * EI + w2 * PI * sigma + w3 * sigma) / cost
    # This balances exploitation (EI), informed exploration (PI * sigma), and pure exploration (sigma)
    
    w_ei = 1.0
    w_pi_sigma = 0.5
    w_sigma = 0.1
    
    scores = (w_ei * ei + w_pi_sigma * pi * sigma_safe + w_sigma * sigma_safe) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
