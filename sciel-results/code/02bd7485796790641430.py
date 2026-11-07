import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement (EI) and Uncertainty (UCB)
    weighted by cost, with a time-based fallback.
    
    Args:
        mu: Mean of the GP prediction, shape (2000,)
        sigma: Standard deviation of the GP prediction, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each point, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better.
    """
    # Time guard: if we are close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        
        scores = ei / cost_safe
        scores[~np.isfinite(scores)] = 0.0
        return scores

    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety: clip sigma and cost to avoid division by zero or inf
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) for minimization:
    # We want to minimize, so we look for low values.
    # UCB = mu - beta * sigma (lower bound of the confidence interval)
    # To make it an acquisition function (higher is better), we use -UCB or similar.
    # However, standard UCB for minimization is often defined as:
    # score = -mu + beta * sigma (we prefer low mu and high sigma)
    # Let's use a standard exploration term.
    # beta is a trade-off parameter. 1.0 is a common starting point.
    beta = 1.0
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Cost-aware UCB
    # Normalizing UCB is tricky because it can be negative.
    # Let's just divide by cost.
    score_ucb = ucb_score / cost_safe
    
    # Combine EI and UCB
    # EI is generally more robust for exploitation.
    # UCB helps with exploration.
    # We can blend them.
    # A simple linear combination:
    # weights: 0.8 for EI, 0.2 for UCB?
    # Or just use EI as the primary and add a small UCB term?
    # Let's try: score = score_ei + 0.1 * score_ucb
    # But score_ucb can be negative if mu is high.
    # Let's ensure non-negative scores for stability?
    # Actually, the relative ranking is what matters.
    
    # Let's try a different approach:
    # Use EI as the main driver, but boost points with high uncertainty
    # if they are not too expensive.
    
    # Alternative: Use PI (Probability of Improvement) as a secondary signal?
    # PI = Phi(z)
    # score_pi = PI / cost_safe
    
    # Let's stick to EI and UCB.
    # To avoid negative scores dominating, we can normalize or shift.
    # But simple division by cost is the key constraint.
    
    # Let's try: scores = score_ei + 0.05 * score_ucb
    # This adds a small exploration bonus.
    
    scores = score_ei + 0.05 * score_ucb
    
    # Ensure finite outputs
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
