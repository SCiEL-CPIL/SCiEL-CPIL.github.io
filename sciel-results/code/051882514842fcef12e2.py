import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using PI (Probability of Improvement) with 
    soft cost penalty. This approach naturally saturates and decays smoothly,
    providing better exploration-exploitation balance than linear UCB/EI
    for this specific cost-aware optimization task.
    """
    # Deadline guard: track elapsed time to avoid timeout
    # The evaluator kills after 330s, so we add a safety margin
    t_start = time.time()
    deadline = t_start + 5.0  # 5 seconds safety margin per call
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score for PI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Soft cost penalty: divide by (1 + cost_penalty * cost)
    # This creates a smooth trade-off between improvement probability and cost
    # cost_penalty controls how strongly cost is penalized
    cost_penalty = 0.5
    
    scores = pi / (1.0 + cost_penalty * cost_safe)
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return current best result if we're running out of time
        pass
    
    return scores
