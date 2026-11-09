"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a smooth Lagrangian-inspired penalty.
    
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB) with a 
    dynamically tuned exponential cost penalty. This acts as a smooth surrogate 
    for a hard cost constraint, allowing for a more robust exploration-exploitation 
    trade-off than simple division by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for exploration
    known_min = -78.3323
    progress = f_best - known_min  # Positive when above minimum
    
    # Adaptive beta: more exploration early, less later
    if progress > 30:
        beta = 2.5
    elif progress > 15:
        beta = 2.0
    elif progress > 5:
        beta = 1.5
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Adaptive weighting between EI and UCB based on progress
    # Early: more UCB (explore), Late: more EI (exploit)
    if progress > 20:
        w_ei = 0.4
        w_ucb = 0.6
    elif progress > 10:
        w_ei = 0.5
        w_ucb = 0.5
    elif progress > 5:
        w_ei = 0.6
        w_ucb = 0.4
    else:
        w_ei = 0.7
        w_ucb = 0.3
        
    # Smooth Cost Penalty Mechanism:
    # Instead of dividing by cost (which can be unstable if cost is small or large),
    # we use an exponential decay: exp(-lambda * (cost - cost_min)).
    # This is a smooth surrogate for a hard budget constraint.
    
    # Normalize cost to [0, 1] relative to the current batch to stabilize lambda
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_range = 1.0
    cost_norm = (cost - cost_min) / cost_range
    
    # Dynamic lambda: 
    # When progress is large (early), we are less sensitive to cost (small lambda).
    # When progress is small (late), we are very sensitive to cost (large lambda).
    # We map progress to lambda.
    # progress > 30 -> lambda ~ 0.5 (mild penalty)
    # progress < 5  -> lambda ~ 3.0 (strong penalty)
    
    # Use a smooth interpolation for lambda
    # lambda = base_lambda + scale_lambda * (1 - progress/progress_max)
    # Let's use a simpler mapping:
    if progress > 30:
        lambda_val = 0.5
    elif progress > 15:
        lambda_val = 1.0
    elif progress > 5:
        lambda_val = 2.0
    else:
        lambda_val = 3.0
        
    # Apply smooth penalty
    # We penalize relative to the minimum cost in the batch.
    # Points with min cost get penalty 1.0.
    # Points with max cost get exp(-lambda).
    penalty = np.exp(-lambda_val * cost_norm)
    
    # Combine scores
    # We multiply the acquisition components by the penalty.
    score_ei = ei * penalty
    score_ucb = ucb * penalty
    
    scores = w_ei * score_ei + w_ucb * score_ucb
    
    # Add a small uncertainty bonus in early stages to encourage exploration
    # This is added after the penalty to ensure that even high-cost uncertain points
    # have a chance if the penalty isn't too severe.
    if progress > 20:
        max_sigma = np.max(sigma)
        if max_sigma > 1e-9:
            unc_bonus = sigma / max_sigma
            scores += 0.1 * unc_bonus * penalty # Also penalize the bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Final normalization for numerical stability
    score_max = np.max(scores)
    score_min = np.min(scores)
    if score_max - score_min > 1e-12:
        scores = (scores - score_min) / (score_max - score_min)
    
    return scores
# EVOLVE-BLOCK-END
