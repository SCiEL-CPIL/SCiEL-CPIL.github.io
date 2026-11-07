import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a stochastic knapsack-inspired score: P(improvement) * E[improvement] / cost
    which is equivalent to expected improvement per unit cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = 300.0  # seconds, with safety margin before 330s kill
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # The stochastic knapsack formulation:
    # Score = P(mu < f_best) * E[f_best - mu | mu < f_best] / cost
    # This is equivalent to EI / cost
    # EI already incorporates the probability of improvement and the expected magnitude
    
    # Cost-aware score: EI divided by cost
    # This directly maximizes expected regret reduction per unit cost
    scores = ei / cost_safe
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Check time
    if time.time() - start_time > deadline * 0.8:
        return scores
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima by slightly favoring uncertain points
    # but scaled down so it doesn't dominate the cost-aware EI
    
    # Normalize sigma to get a relative uncertainty measure
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-10:
        rel_uncertainty = sigma_safe / sigma_mean
    else:
        rel_uncertainty = np.ones_like(sigma_safe)
    
    # Small exploration term: sqrt of relative uncertainty, weighted down
    # and also divided by cost to maintain cost-awareness
    exploration_weight = 0.15
    exploration_bonus = exploration_weight * np.sqrt(rel_uncertainty) / cost_safe
    
    # Combine: primary score is cost-aware EI, with small exploration bonus
    final_scores = scores + exploration_bonus
    
    # Final numerical safety
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(final_scores)):
        final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
