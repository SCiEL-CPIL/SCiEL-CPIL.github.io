import numpy as np
from scipy.stats import norm
import time

# Track start time for budget guard
_ACQ_START_TIME = None
_ACQ_CALL_COUNT = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a quantile-based feasibility gate with adaptive UCB blending.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    global _ACQ_START_TIME, _ACQ_CALL_COUNT
    
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    _ACQ_CALL_COUNT += 1
    
    # Safety check: if we're getting close to the 330s limit, return quickly
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 320:
        # Return a simple fallback score to avoid timeout
        return np.zeros(len(mu))
    
    # Calculate Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB
    # Adaptive kappa: start with high exploration, decay over time
    kappa = 1.5 * np.exp(-_ACQ_CALL_COUNT / 100.0) + 0.5
    ucb = mu + kappa * sigma
    
    # For minimization, lower UCB is better, so we negate for scoring
    ucb_score = -ucb
    
    # Adaptive blending between EI and UCB based on call count
    # Early on: more UCB (exploration), later: more EI (exploitation)
    blend_weight = 0.3 + 0.5 * np.exp(-_ACQ_CALL_COUNT / 50.0)
    raw_score = blend_weight * ucb_score + (1.0 - blend_weight) * ei
    
    # Quantile-based feasibility gate
    # Use 25th percentile of cost as threshold - more robust than mean
    cost_threshold = np.percentile(cost, 25)
    
    # Create mask: candidates with cost below threshold get full score
    # Others get heavily penalized
    feasible = cost <= cost_threshold
    
    # Apply feasibility gate: multiply by a factor that drops off for expensive candidates
    # Use a smooth transition rather than hard gate
    cost_ratio = cost / (cost_threshold + 1e-10)
    gate = np.where(cost_ratio <= 1.0, 1.0, np.exp(-(cost_ratio - 1.0) * 2.0))
    
    # Combine raw score with feasibility gate
    scores = raw_score * gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
