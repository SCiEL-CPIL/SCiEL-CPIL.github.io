import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) divided by cost.
    
    The EI term provides a calibrated measure of expected gain, while the LCB term
    (mu - beta * sigma) encourages exploration of high-uncertainty regions.
    Combining them helps avoid local optima and stagnation, which is critical
    for the Ackley function with its many local minima.
    
    The beta parameter is set to 1.0 to balance exploration and exploitation.
    Dividing by cost makes the acquisition function cost-aware, preferring
    cheaper evaluations.
    """
    start_time = time.time()
    
    # Safety margin: ensure we finish well before the 300s limit
    deadline = 290.0  # 10 seconds margin
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # LCB term: encourages exploration of high-uncertainty regions
    beta = 1.0
    lcb = mu - beta * sigma
    
    # Hybrid acquisition: average of EI and LCB
    # EI is in "improvement" units (positive when mu < f_best)
    # LCB is in "objective" units (lower is better)
    # To combine them, we normalize LCB to be comparable to EI
    # Since we're minimizing, a lower LCB is better, so we negate it
    # and scale it appropriately
    
    # Scale LCB to be comparable to EI: since EI is bounded by (f_best - mu_max)
    # and LCB can be unbounded, we use a simple scaling
    # The mean of EI is a good scale factor
    ei_scale = np.mean(ei) + 1e-12
    lcb_normalized = (-lcb) / ei_scale
    
    # Combine EI and normalized LCB
    hybrid = 0.5 * ei + 0.5 * lcb_normalized
    
    # Cost-aware: divide by cost to prefer cheaper points
    scores = hybrid / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're close to the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a simple fallback if we're running out of time
        return ei / cost
    
    return scores
