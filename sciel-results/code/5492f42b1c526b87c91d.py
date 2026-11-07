"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    combines cost-normalized EI with an intrinsic uncertainty-based
    exploration term that adapts to the GP's confidence landscape.
    
    The key structural difference from pure EI/cost is the use of a
    log-scaled uncertainty bonus that is independent of EI, allowing
    the algorithm to explore regions where the GP is uncertain even
    if current EI is low (e.g., near the optimum or in unexplored
    basins). The cost model is handled via a multiplicative penalty
    rather than simple division, which better captures the trade-off.
    
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
    t_start = time.time()
    deadline = 300.0 - 5.0  # 5 second safety margin
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: use a softer cost model
    # Instead of ei/cost, use ei / (cost^alpha) with alpha < 1
    # This reduces the penalty for expensive points, allowing exploration
    alpha = 0.5
    ei_cost = ei / (cost_safe ** alpha)
    
    # Normalize EI component
    ei_max = np.max(ei_cost)
    if ei_max > 1e-12:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros_like(ei_cost)
    
    # Intrinsic exploration bonus: based on predictive uncertainty
    # Use log-scaled sigma to reward high-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-8:
        # Log-scale: compresses the range, making the bonus smoother
        sigma_log = np.log1p(sigma_safe) / np.log1p(sigma_max)
    else:
        sigma_log = np.zeros_like(sigma_safe)
    
    # Adaptive exploration weight: higher when EI is concentrated
    # (i.e., when we're close to convergence)
    ei_spread = np.std(ei_norm)
    if ei_spread < 0.1:
        # Near convergence: explore more
        exploration_weight = 0.6
    elif ei_spread < 0.3:
        exploration_weight = 0.4
    else:
        # Early stage: rely more on EI
        exploration_weight = 0.2
    
    # Combine EI and exploration
    # Use a convex combination with adaptive weight
    scores = (1.0 - exploration_weight) * ei_norm + exploration_weight * sigma_log
    
    # Additional structural element: penalize points where the GP is
    # confident but not good (low sigma, high mu) to avoid re-evaluating
    # known-bad regions
    confidence_penalty = np.where(sigma_safe < np.percentile(sigma_safe, 25), 
                                  1.0 - np.minimum(ei_norm, 0.5), 1.0)
    scores *= confidence_penalty
    
    # Deadline check
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Final safety: ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
