# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _check_deadline():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses a combination of Expected Improvement and Upper Confidence Bound
    to balance exploration and exploitation.
    """
    # Guard against division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization:
    # We want to maximize the acquisition score, so we use:
    # UCB = mu - kappa * sigma  (lower mu and higher sigma are better for exploration)
    # But since we're maximizing the acquisition score, and lower objective is better,
    # we can use: score = -mu + kappa * sigma
    # Actually, let's think about this more carefully.
    # For minimization, UCB typically is: mu - kappa * sigma (we prefer low mu, high sigma)
    # But our acquisition function should return higher scores for more promising points.
    # So for minimization, we want to pick points with low mu (good) and high sigma (uncertain).
    # A common approach is: acquisition = -mu + kappa * sigma
    # Or equivalently, we can normalize and combine.
    
    # Let's use a weighted combination. First, let's normalize both to be in similar scales.
    # EI is already non-negative and represents expected improvement.
    # For UCB in minimization context, we use: -mu + kappa*sigma
    # But this can be negative. Let's instead use a different formulation.
    
    # Alternative: Use PI (Probability of Improvement) and EI
    # Or use a Gaussian process-based approach with a tuned kappa.
    
    # Let's try: acquisition = w1 * EI + w2 * (-mu + kappa * sigma)
    # But we need to be careful about scales. Let's normalize.
    
    # Actually, a cleaner approach for minimization:
    # The "confidence bound" for minimization is: mu - kappa * sigma
    # We want to maximize this, so points with low mu and high sigma score high.
    # But mu - kappa*sigma can be very negative. Let's shift it.
    
    # Let's use a different strategy: 
    # 1. Compute EI
    # 2. Compute a "novelty" or "exploration" term based on sigma
    # 3. Combine them
    
    # A well-known hybrid is to use EI with a small exploration bonus.
    # Let's try: acquisition = EI + epsilon * sigma
    # where epsilon is small, like 0.1
    
    # Or better: use the fact that EI already incorporates both exploitation (via f_best - mu)
    # and exploration (via sigma). Pure EI might be sufficient, but let's try adding
    # a small UCB component to encourage more exploration early on.
    
    # Let's use: acquisition = EI + 0.1 * sigma
    # This adds a small exploration bonus proportional to uncertainty.
    
    epsilon = 0.1
    ucb_term = -mu + 0.5 * sigma  # For minimization, we prefer low mu, high sigma
    
    # Normalize both terms to have similar scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term) + 1e-9
    ucb_norm = (ucb_term - ucb_mean) / ucb_std
    
    # Weighted combination
    w_ei = 0.7
    w_ucb = 0.3
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
