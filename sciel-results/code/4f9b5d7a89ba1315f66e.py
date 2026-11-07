import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Levy 2D.
    
    Strategy: Combine Expected Improvement (exploitation) with a strong
    cost penalty to bias search toward the low-cost region where the
    best value/cost trade-off is likely. The Levy function has its
    global minimum at (0,0) in the original space, but shifted instances
    move it to a random point. Cost = 1 + mean(x_scaled) where x_scaled
    maps [-10,10] to [0,1]. So low cost corresponds to low x values
    (left/bottom of the domain).
    
    We use a weighted combination:
    - EI term: rewards points where mu < f_best
    - Cost penalty: divide by cost^2 to strongly favor low-cost regions
    - Small exploration bonus for high sigma to ensure coverage
    
    This is a structural change from pure UCB/cost: we anchor to f_best
    via EI which provides a natural scale, then apply a quadratic cost
    penalty to strongly bias toward the cheap corner while still allowing
    the GP to guide local search.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(-z) + sigma * phi(z) where z = (f_best - mu)/sigma
    diff = f_best - mu
    z = diff / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = diff * Phi(-z) + sigma * phi(z)
    # Note: Phi(-z) = 1 - Phi(z)
    ei = diff * (1.0 - cdf) + sigma * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: divide by cost^2 for strong bias toward low cost
    # Add a small exploration term proportional to sigma/cost to ensure
    # we don't completely ignore uncertain regions
    exploration = 0.3 * sigma / cost
    
    scores = (ei + exploration) / (cost * cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
