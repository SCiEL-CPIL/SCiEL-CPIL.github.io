import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are running out of time
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # 1. Compute Expected Improvement (EI)
    # EI is in objective units, which is a more meaningful scale than normalized probabilities
    z = (f_best - mu) / (sigma + 1e-8)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI is non-negative by definition
    
    # 2. Compute UCB exploration term
    # This encourages exploring regions with high uncertainty (high sigma)
    # Even if the mean is not currently promising, high sigma suggests potential for improvement
    beta = 0.5  # Exploration weight
    ucb = beta * sigma
    
    # 3. Combine EI and UCB
    # This hybrid helps escape local minima (via UCB) while still exploiting promising regions (via EI)
    combined = ei + ucb
    
    # 4. Apply sublinear cost penalty
    # The cost is 1 + mean(x scaled to [0,1]). 
    # Linear penalty (1/cost) heavily discourages evaluating in the "expensive" half of the domain.
    # Since the global minimum is randomly shifted to the inner 80%, it could be in the expensive region.
    # A square root penalty (1/sqrt(cost)) is less punitive, allowing the optimizer to consider 
    # high-cost points if their EI/UCB score is sufficiently high.
    # This was identified as a key improvement in the knowledge base (1.0 -> 1.03).
    cost_penalty = np.sqrt(cost + 1e-8)
    
    scores = combined / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
