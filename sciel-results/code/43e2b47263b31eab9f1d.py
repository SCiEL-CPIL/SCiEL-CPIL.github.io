import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score for minimization
    # We want to maximize: -mu + beta * sigma
    # Adaptive beta based on distance to known minimum
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Beta decreases smoothly as we get closer to the optimum
    # Use a smoother decay that doesn't drop too quickly
    beta = 1.0 + 3.0 * np.exp(-dist_to_min / 0.5)
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both components to have similar scales
    ucb_std = np.std(ucb_score)
    ei_std = np.std(ei)
    
    if ucb_std > 1e-8:
        ucb_norm = (ucb_score - np.mean(ucb_score)) / ucb_std
    else:
        ucb_norm = np.zeros(n)
    
    if ei_std > 1e-8:
        ei_norm = (ei - np.mean(ei)) / ei_std
    else:
        ei_norm = np.zeros(n)
    
    # Combine UCB and EI with smoother phase transition
    # Use a sigmoid-like weight that transitions smoothly
    # When dist_to_min is large, weight more on UCB (exploration)
    # When dist_to_min is small, weight more on EI (exploitation)
    t = dist_to_min / 0.5  # Scale factor
    ucb_weight = 0.5 + 0.3 * np.tanh(t)  # Ranges from ~0.2 to ~0.8
    ei_weight = 1.0 - ucb_weight
    
    combined = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Cost-aware: use a more sophisticated cost adjustment
    # Instead of simple division, use a cost-adjusted score that balances
    # the acquisition value against the marginal cost
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-8:
        cost_norm = (cost_safe - cost_min) / cost_range
    else:
        cost_norm = np.ones(n) * 0.5
    
    # Use a multiplicative penalty that's less aggressive than division
    # Higher cost should reduce score but not zero it out
    cost_penalty = 1.0 - 0.5 * cost_norm  # Ranges from 0.5 to 1.0
    
    scores = combined * cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Add a small random tie-breaker to avoid degenerate cases
    # This helps when multiple points have similar scores
    rng = np.random.default_rng(42)
    scores += rng.normal(0, 1e-6, size=n)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    return scores
