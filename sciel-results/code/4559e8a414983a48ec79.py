"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid UCB + EI approach with adaptive exploration weight
    that increases early in the budget and decreases as we approach
    the cost limit, balancing exploration and exploitation for
    multimodal functions like Shekel 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - higher is better
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration: mu - beta * sigma (for minimization, lower is better)
    # We want to maximize the acquisition, so we use: -mu + beta * sigma
    # The negative mu term favors low predicted values (good for minimization)
    # The beta * sigma term favors high uncertainty (exploration)
    
    # Adaptive beta: start high for exploration, decrease as budget runs out
    # Estimate progress through the budget based on cost spent
    # Total cost limit is 30, initial 5 points cost ~5*1.5=7.5, so remaining ~22.5
    # cost array gives per-point cost, sum of evaluated costs tells us progress
    # But we don't have that directly - use heuristic based on f_best improvement
    
    # Use a simpler adaptive strategy: beta decreases with the number of
    # "effective evaluations" estimated from the cost structure
    # Higher cost points are more informative, so weight by inverse cost
    
    # Adaptive beta: interpolate between high exploration early and low later
    # Use the mean sigma as a proxy for remaining uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = max(sigma_max - sigma_min, 1e-8)
    
    # Normalized uncertainty: 0 = all certain, 1 = all uncertain
    uncertainty_level = sigma_mean / (sigma_mean + 0.1 * sigma_range + 1e-8)
    
    # Beta: higher when uncertainty is high (early), lower when certain (late)
    # Range: ~0.5 to ~2.0
    beta = 0.5 + 1.5 * uncertainty_level
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-normalize both components
    ei_cost = ei / cost
    ucb_cost = ucb_score / cost
    
    # Hybrid: blend EI and UCB
    # EI is more exploitative, UCB is more exploratory
    # Weight based on relative confidence
    # When EI is large relative to sigma, we're confident -> favor EI
    # When EI is small relative to sigma, we're uncertain -> favor UCB
    
    # Confidence metric: how much improvement do we expect vs uncertainty
    confidence = ei / (sigma_safe + 1e-8)
    
    # Sigmoid blending: high confidence -> EI, low confidence -> UCB
    # Scale confidence to get a good sigmoid range
    blend_weight_ei = 1.0 / (1.0 + np.exp(-0.5 * confidence))
    blend_weight_ucb = 1.0 - blend_weight_ei
    
    # Base score: weighted blend of EI and UCB, both cost-normalized
    score = blend_weight_ei * ei_cost + blend_weight_ucb * ucb_cost
    
    # Add a small pure exploration bonus for very high-uncertainty regions
    # This helps escape local optima in multimodal functions
    # Scale by cost to maintain cost-awareness
    # Use a softplus-like function to avoid over-penalizing moderate uncertainty
    exploration_bonus = np.log1p(sigma_safe) / (np.log1p(sigma_mean) + 1e-8)
    score += 0.03 * exploration_bonus / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
