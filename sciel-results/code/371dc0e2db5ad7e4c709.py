"""Cost-aware acquisition for Bayesian optimization of Griewank 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for speed
# We use a combination of Probability of Improvement (PI) and Expected Improvement (EI)
# to balance exploitation and exploration, normalized by cost.

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, best observed value so far
    cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using scipy.stats.norm would be slow if imported every time, but we can use
    # a fast approximation or precomputed table. However, for 2000 points, 
    # using a vectorized normal CDF is fine if we import it once at module level.
    # Let's import norm at the top level for speed.
    
    # We'll use a hybrid: PI is bounded and smooth, EI is unbounded.
    # A common cost-aware approach is to divide by cost.
    # Let's try: (PI + alpha * EI) / cost
    # Or simply PI / cost, which was found to improve in the knowledge base.
    
    # From knowledge: "Switching from EI/sqrt(cost) to PI/cost changes the exploration-exploitation balance. 
    # PI is bounded and sigmoidal, avoiding the unbounded exploration of EI in high-uncertainty regions. 
    # Measured: 1.132 -> 1.803."
    
    # So let's use PI / cost.
    
    # But wait, the current program uses EI / cost.
    # Let's implement PI / cost as the primary strategy.
    
    # Normal CDF: 0.5 * (1 + erf(z / sqrt(2)))
    # Using numpy's erf for speed
    pi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    
    # Cost-aware score
    scores = pi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
