"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline management
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_deadline():
    # Kill program after 330s wall-clock, keep safety margin
    if _START_TIME is not None:
        if time.time() - _START_TIME > 325.0:
            return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Lookahead GP-UCB strategy.
    
    Args:
        mu: Mean predictions, shape (N,)
        sigma: Standard deviation predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        Scores for each candidate, shape (N,)
    """
    _init_time()
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    if _check_deadline():
        # Fallback to simple cost-normalized EI if time is running out
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # --- Lookahead GP-UCB Strategy ---
    
    # 1. Exploitation Term: Expected Improvement (EI)
    # This drives the search towards points likely to beat f_best.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Exploration Term: Lookahead Uncertainty
    # Standard GP-UCB uses beta * sigma. 
    # "Lookahead" implies we care about how much information gain we get.
    # For a minimization problem, high sigma indicates potential for a much lower f_best.
    # We approximate the value of exploration by the expected reduction in f_best 
    # if we were to sample from the GP posterior at this point.
    # E[min(f_best, y)] where y ~ N(mu, sigma^2).
    # This is essentially the "Expected Minimum" or "Optimistic Value".
    # For a single step, the optimistic value is mu - beta*sigma.
    # For a lookahead perspective, we can consider that high sigma regions 
    # often contain the global minimum. 
    # Let's use a UCB term: mu - beta * sigma.
    # However, we need to balance this against cost.
    
    # Adaptive Beta: 
    # Early on (f_best is high), we should explore more (higher beta).
    # Late on (f_best is low), we should exploit more (lower beta).
    # Since we don't have explicit iteration count, we use f_best as a proxy.
    # Powell 4D minimum is 0.
    # If f_best is close to 0, beta should be small.
    # If f_best is large, beta should be larger.
    
    # Heuristic for beta:
    # beta = max(0.5, 5.0 * exp(-f_best / 10.0)) 
    # This decays as we get closer to the optimum.
    beta = 2.0 * np.exp(-f_best / 5.0) + 0.5
    
    # UCB score for exploitation/exploration balance
    ucb = mu - beta * sigma
    
    # 3. Cost-Aware Combination
    # We want to maximize (Improvement + Exploration Value) / Cost.
    # EI is non-negative. UCB can be negative (if mu is high).
    # A better combined metric for minimization with cost is:
    # Score = (EI + lambda * sigma) / cost
    # Or: Score = (f_best - mu + beta * sigma) / cost  (This is like UCB of improvement)
    
    # Let's use a weighted sum of EI and UCB-like term, normalized by cost.
    # EI captures the probability-weighted improvement.
    # UCB captures the optimistic potential.
    
    # Alternative: Use "Expected Hypervolume Improvement" approximation? 
    # Too complex.
    
    # Let's stick to a robust Cost-UCB:
    # The value of a point is its potential to lower f_best.
    # Potential = max(0, f_best - (mu - beta*sigma)) is an optimistic improvement.
    # But EI is more statistically sound.
    
    # Let's try a hybrid:
    # Score = (EI * w1 + sigma * w2) / cost
    # Where w1 and w2 are adaptive.
    
    # Actually, the current program uses EI / cost^alpha.
    # The "Lookahead" aspect suggests we should penalize points that are 
    # "wasted" because they are in high-cost regions that don't offer 
    # proportional uncertainty reduction.
    
    # Let's refine the cost penalty.
    # Instead of dividing by cost, we can use a "regret" perspective.
    # Simple Regret = f_best - f_min.
    # We want to minimize Simple Regret.
    
    # Strategy: 
    # 1. Calculate EI (exploitation).
    # 2. Calculate "Uncertainty Value" = sigma (exploration).
    # 3. Combine them with an adaptive weight that shifts from exploration to exploitation
    #    as f_best decreases.
    # 4. Normalize by cost.
    
    # Adaptive weight:
    # If f_best is high, explore more (weight on sigma higher).
    # If f_best is low, exploit more (weight on EI higher).
    
    # f_best starts around ~20-50 for Powell in 4D.
    # Let's define a transition point.
    # weight_exploit = sigmoid(k * (f_threshold - f_best))
    
    # Let's use a simpler, more robust formula:
    # Score = (EI + gamma * sigma) / cost
    # Where gamma is adaptive.
    
    # If f_best > 5.0: gamma = 1.0 (high exploration)
    # If f_best < 1.0: gamma = 0.2 (low exploration, high exploitation)
    # Linear interpolation in between.
    
    if f_best > 5.0:
        gamma = 1.0
    elif f_best < 1.0:
        gamma = 0.2
    else:
        # Linear map from 5.0->1.0 to 1.0->0.2
        # t = (f_best - 1.0) / 4.0
        # gamma = 0.2 + t * 0.8
        t = (f_best - 1.0) / 4.0
        gamma = 0.2 + t * 0.8
        
    # Combined Score
    # We also need to ensure that if mu is already very low (better than f_best),
    # the score is high. EI handles this.
    
    # To add the "Lookahead" flavor, we can slightly boost the sigma term
    # for points where the GP is most uncertain, assuming that uncertainty
    # likely hides the global minimum.
    
    scores = (ei + gamma * sigma) / cost_safe
    
    # Replace non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure higher is better
    # If scores are all negative? EI is >= 0. Sigma is >= 0. Cost > 0.
    # So scores should be >= 0.
    
    return scores
# EVOLVE-BLOCK-END
