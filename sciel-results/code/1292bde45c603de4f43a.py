"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track time and potentially adapt parameters if needed
# However, the function signature is stateless, so we rely on f_best and cost for context.
# We will use a fixed, well-tuned structural approach.

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Cost-Weighted Probability of Improvement (CWPI)
    with an Uncertainty Bonus.
    
    This structural change replaces Expected Improvement (EI) with Probability of Improvement (PI).
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    Why PI over EI?
    EI is sensitive to the magnitude of improvement. In multi-modal landscapes like Shekel,
    EI can sometimes favor points with high variance but low mean improvement if the mean is far,
    or fail to move decisively towards a local minimum if the probability of improvement is moderate
    but the expected gain is small. PI focuses on the likelihood of *any* improvement, which
    combined with the cost penalty, encourages cheap, reliable steps towards lower values.
    
    We add a small exploration term proportional to sigma to prevent premature convergence
    to a single local minimum, scaled by cost to ensure we don't waste budget exploring
    expensive regions.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for non-finite values
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # 1. Calculate Probability of Improvement (PI)
    # z-score for the probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf (though norm.cdf is stable, 
    # extreme values can lead to 0 or 1, which is fine, but let's be safe)
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    # 2. Calculate Exploration Bonus
    # We want to explore regions with high uncertainty, but penalize by cost.
    # Normalize sigma to [0, 1] range relative to the max sigma in the batch
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # 3. Combine PI and Exploration
    # Weight of exploration. 
    # Early on, we might want more exploration, but f_best is the only proxy for "progress".
    # As f_best decreases, we should exploit more.
    # Shekel min is ~ -10.5. Start is ~ 0.
    # Let's use a fixed weight for the exploration term, tuned for Shekel.
    # A small weight ensures we don't just pick random points.
    exploration_weight = 0.15
    
    # Raw score components
    # We divide by cost to make it cost-aware.
    # PI is in [0, 1]. Cost is in [1, 2].
    # So PI/cost is in [0, 1].
    # sigma_norm is in [0, 1].
    
    score_pi = pi / cost_safe
    score_exp = exploration_weight * sigma_norm / cost_safe
    
    scores = score_pi + score_exp
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
