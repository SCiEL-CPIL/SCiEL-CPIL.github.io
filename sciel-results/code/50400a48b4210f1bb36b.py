# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) with Entropy Bonus.
    
    This strategy uses the exact Expected Improvement formula for a Gaussian 
    posterior, adjusted for evaluation cost. Unlike UCB, EI is theoretically 
    grounded in the probability of improvement and is more stable in the 
    presence of noise. We divide by cost to maximize expected improvement 
    per unit cost. We also add a small entropy bonus to encourage exploration 
    in regions of high predictive uncertainty, which is crucial for multi-modal 
    functions like Styblinski-Tang.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Standardize f_best for numerical stability in Z calculation
    # Z = (f_best - mu) / sigma
    # If mu is very close to f_best, Z can be unstable, but sigma > 1e-8 prevents div by 0.
    Z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is the CDF and phi is the PDF of the standard normal.
    # We use scipy.stats.norm for accuracy.
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical errors can sometimes make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: We want to maximize EI per unit cost.
    # Dividing by cost penalizes expensive evaluations (edges of the domain).
    # This effectively biases the search towards the center where cost is lower 
    # (cost = 1 + mean(x_scaled), min cost ~1.0 at center, max ~2.0 at corners? 
    # Actually x_scaled in [0,1], mean is 0.5, so cost is 1.5 on average. 
    # Wait, cost = 1 + mean(x_scaled). For a single point, x_scaled is a vector? 
    # The prompt says "mean(x scaled to [0, 1])". This likely means the mean of the coordinates.
    # So cost ranges from 1.0 (center) to 2.0 (corners) roughly? 
    # If x is in [-5,5], scaled to [0,1], center is 0.5,0.5 -> cost 1.5. 
    # Corners are 0,0 or 1,1 -> cost 1.0 or 2.0? 
    # Let's assume cost is proportional to distance from center or similar.
    # Regardless, dividing by cost is a standard cost-aware adjustment.
    cai = ei / cost
    
    # Entropy Bonus:
    # Add a small term proportional to the predictive entropy to encourage exploration.
    # Entropy of N(mu, sigma^2) is 0.5 * log(2*pi*e*sigma^2).
    # We only need the sigma part for relative ranking.
    # H = 0.5 * log(sigma^2) = log(sigma)
    # We scale this down significantly so it doesn't dominate EI.
    # If sigma is small, log(sigma) is large negative. We want high sigma to be good.
    # So we use -log(sigma) or just sigma? 
    # Actually, high sigma means high uncertainty. 
    # Let's use a small multiple of sigma itself, or a transformed entropy.
    # A common heuristic is to add a small epsilon * sigma.
    # But EI already has a sigma term. 
    # Let's try a very small entropy bonus: 0.01 * sigma.
    # This ensures that if EI is zero (mu < f_best is unlikely for candidates? No, mu can be > f_best),
    # we still explore uncertain regions.
    # Actually, if mu < f_best, EI > 0. If mu > f_best, EI can still be > 0 if sigma is large enough.
    # The entropy bonus helps when EI is very small but uncertainty is high.
    
    entropy_bonus = 0.01 * sigma
    
    scores = cai + entropy_bonus
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
